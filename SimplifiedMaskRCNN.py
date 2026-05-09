import numpy as np
import math
import os
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import lightning as L
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from modules import ViTDetBackbone, TorchvisionSwinV2Backbone, ResNet50Backbone, BackboneWithFPN
from modules import AltTorchvisionResNet50Backbone, AltTorchvisionSwinV2Backbone
from dataset import OAMTCDCOCODataset, get_train_transforms
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.callbacks.finetuning import BaseFinetuning
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import write_png
from tifffile import imread
import torch.nn as nn
import torchvision.transforms.functional as F
import argparse
import sys

# TODO: Move seed_everything to main.
# TODO: Add seed options as args in dict form {s1: 1234, s2: 6147, s3: 4319}
L.seed_everything(1234)


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def collate_fn(batch):
    return tuple(zip(*batch))


# TODO: For checkpoints, instead of giving it as an argument use a dict with key: arch_type, value: ckpt path
# TODO: Remove the use_pretrained flag since we're not doing supervised
# TODO: remove use_lora flag as it should be a strategy
# TODO: change strategy choices to ['gradualLLRD', 'full', 'frozen', 'lora', 'lora_llrd']
# TODO: add the seed arg
# TODO: remove fast option
# TODO: add lr_decay arg with default 0.75
# TODO: remove warmup up steps arg its hardcoded.
def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--fast", action="store_true", default=False)
    parser.add_argument("--use_pretrained", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str,
                        default="")
    # parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--strategy", type=str,
                        default="full", choices=["gradual", "adaptive", "frozen", "full", "lora"])
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--candidate_path", type=str, default="")

    parser.add_argument("--train_folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--val_folds", type=int, nargs="+", default=[0])

    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--arch_type", type=str, default="")
    return parser.parse_args()


class LitMaskRCNN(L.LightningModule):
    def __init__(self, backbone, train_dataset, val_dataset, test_dataset, args):
        super().__init__()
        # self.save_hyperparameters(ignore=['backbone'])

        self.backbone = backbone
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)
        self.test_sampler = SequentialSampler(self.test_dataset)

        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers

        self.model = self.build_mask_rcnn(self.backbone, self.args.num_classes, self.args.img_size, self.args.fast)
        if self.args.strategy in ["gradual", "frozen"]:
            if self.args.arch_type == "vit":
                for param in self.model.backbone.vit.vit.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.backbone.body.parameters():
                    param.requires_grad = False

        self.val_map_bbox = MeanAveragePrecision(iou_type="bbox")
        self.test_map_bbox = MeanAveragePrecision(iou_type="bbox")
        self.test_map_segm = MeanAveragePrecision(iou_type="segm")
        self.candidate_path = self.args.candidate_path

    @property
    def rpn(self):
        return self.model.rpn

    @property
    def roi_heads(self):
        return self.model.roi_heads

    @property
    def backbone(self):
        # TODO: Add ConvNext support
        arch = self.args.arch_type
        if arch == 'vit':
            return self.model.backbone.vit.vit.model
        elif arch == 'swin':
            return self.model.backbone.body.body
        else:  # resnet
            return self.model.backbone.body

    @staticmethod
    def build_mask_rcnn(backbone, num_classes, img_size):
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        )

        model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=img_size,
            max_size=img_size,
            box_detections_per_img=150
        )
        return model

    def split_decay(self, params):
        decay = []
        no_decay = []
        for n, p in self.model.named_parameters():
            if id(p) in params:
                if p.ndim == 1 or "bias" in n or "norm" in n or "pos_embed" in n:
                    no_decay.append(p)
                else:
                    decay.append(p)
        return decay, no_decay

    def configure_optimizers(self):
        lr = self.args.lr
        wd = self.args.weight_decay
        lr_decay = self.args.lr_decay




        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=True,
                          persistent_workers=True, shuffle=True, drop_last=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, sampler=self.val_sampler, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=False, drop_last=False,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, sampler=self.test_sampler, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=False, drop_last=False,
                          collate_fn=collate_fn)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(l for l in loss_dict.values())
        self.log_dict(loss_dict, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        for output in outputs:
            if "masks" in output:
                # Thresholding (at 0.5) to convert float probs to bool,
                output["masks"] = (output["masks"] > 0.5).squeeze(1).to(torch.uint8)
        self.val_map_bbox.update(outputs, targets)
        return outputs

    def on_validation_epoch_end(self):
        bbox_results = self.val_map_bbox.compute()
        self.log("val/bbox_mAP_50", bbox_results["map_50"], prog_bar=True)
        self.val_map_bbox.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        # Inference mode logic
        outputs = self.model(images)
        for output in outputs:
            if "masks" in output:
                output["masks"] = (output["masks"] > 0.5).squeeze(1).to(torch.uint8)
        # Update test metric
        self.test_map_bbox.update(outputs, targets)
        self.test_map_segm.update(outputs, targets)

    def on_test_epoch_end(self):
        bbox_results = self.test_map_bbox.compute()
        segm_results = self.test_map_segm.compute()

        self.log("test/bbox_mAP", bbox_results["map"], prog_bar=True)
        self.log("test/bbox_mAP_50", bbox_results["map_50"])
        self.log("test/bbox_mAP_small", bbox_results["map_small"])
        self.log("test/mask_mAP", segm_results["map"], prog_bar=True)
        self.log("test/mask_mAP_50", segm_results["map_50"])
        self.log("test/mask_mAP_small", segm_results["map_small"])

        self.test_map_bbox.reset()
        self.test_map_segm.reset()

        # Print a nice summary to the console
        print("\n" + "=" * 30)
        print(f"FINAL TEST MASK mAP: {segm_results['map']:.4f}")
        print(f"FINAL TEST BBOX mAP: {bbox_results['map']:.4f}")
        print("=" * 30 + "\n")

        candidate_img = imread(self.candidate_path)
        candidate_img = F.to_tensor(candidate_img)

        with torch.no_grad():
            output_list = self.model(candidate_img.unsqueeze(0).to(self.device))
        output = output_list[0]
        save_path = os.path.join(self.args.output_dir, "candidate_img.png")
        self.save_inference_image(candidate_img, output, save_path)

    def save_inference_image(self, img_tensor, pred, save_path, threshold=0.5):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_uint8 = (img_tensor.cpu().clamp(0, 1) * 255).byte()
        scores = pred["scores"]
        keep = scores > threshold

        boxes = pred["boxes"][keep]
        masks = pred["masks"][keep].squeeze(1) > 0.5 if "masks" in pred else None

        result_img = img_uint8
        if boxes.numel() > 0:
            result_img = draw_bounding_boxes(result_img, boxes, colors="red", width=3 )
            if masks is not None and masks.any():
                result_img = draw_segmentation_masks(result_img, masks, alpha=0.5, colors="blue" )
        # ALWAYS save (even if no detections)
        write_png(result_img, save_path)
        print(f"Saved inference image to: {save_path}")