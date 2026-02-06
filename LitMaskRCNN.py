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
from modules import ViTDetBackbone
from dataset import OAMTCDCOCODataset
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import write_png
from tifffile import imread
import torch.nn as nn
import torchvision.transforms.functional as F


def collate_fn(batch):
    return tuple(zip(*batch))


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

        self.model = self.build_mask_rcnn(self.backbone, self.args.num_classes, self.args.img_size)
        # if args.freeze_backbone:
        #     for param in self.model.backbone.vit.vit.parameters():
        #         param.requires_grad = False

        self.val_map_bbox = MeanAveragePrecision(iou_type="bbox", device=self.device)

        self.test_map_bbox = MeanAveragePrecision(iou_type="bbox", device=self.device)
        self.test_map_segm = MeanAveragePrecision(iou_type="segm", device=self.device)
        self.candidate_path = self.args.candidate_path

    @staticmethod
    def build_mask_rcnn(backbone, num_classes, img_size):
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )
        mask_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2,
        )
        model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_pooler,
            min_size=img_size,
            max_size=img_size
        )
        return model

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]

        decay_params = []
        no_decay_params = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "bias" in n or "norm" in n or "pos_embed" in n:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.args.lr)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.args.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          persistent_workers=True,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          sampler=self.val_sampler,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=False,
                          drop_last=False,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          sampler=self.test_sampler,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=False,
                          drop_last=False,
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
        self.log("val/bbox_mAP", bbox_results["map"], prog_bar=True)
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
        labels = pred["labels"][keep]

        result_img = img_uint8

        if boxes.numel() > 0:
            result_img = draw_bounding_boxes(
                result_img,
                boxes,
                colors="red",
                width=3,
            )

            if masks is not None and masks.any():
                result_img = draw_segmentation_masks(
                    result_img,
                    masks,
                    alpha=0.5,
                    colors="blue",
                )
        # ALWAYS save (even if no detections)
        write_png(result_img, save_path)
        print(f"Saved inference image to: {save_path}")