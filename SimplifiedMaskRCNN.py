import numpy as np
import time
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
from utils import write_dict_to_yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from modules import ViTDetBackbone, TorchvisionSwinV2Backbone, ResNet50Backbone, BackboneWithFPN, ConvNeXtBackbone
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


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def collate_fn(batch):
    return tuple(zip(*batch))


def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--node", type=str, default="not specified")  # added to log which node was used for which exp
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=0.75)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--use_pretrained", action="store_true", default=False)
    parser.add_argument("--strategy", type=str,
                        default="full", choices=["frozen", "full", "lora"])
    parser.add_argument("--split", type=str,
                        default="full", choices=["full", "half", "quarter"])
    parser.add_argument("--seed", type=str,
                        default="s1", choices=["s1", "s2", "s3"])
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--arch_type", type=str,
                        default="resnet50", choices=['swin', 'convnext', 'resnet50'])
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
        # self.train_sampler = RandomSampler(self.train_dataset)
        # self.val_sampler = SequentialSampler(self.val_dataset)
        # self.test_sampler = SequentialSampler(self.test_dataset)

        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers

        self.model = self.build_mask_rcnn(self.backbone, self.args.num_classes, self.args.img_size)
        self.configure_training_strategy()

        self.val_map_bbox = MeanAveragePrecision(iou_type="bbox")
        self.test_map_bbox = MeanAveragePrecision(iou_type="bbox")
        self.test_map_segm = MeanAveragePrecision(iou_type="segm")
        self.dict_of_metrics = {}

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

    @staticmethod
    def no_weight_decay(name, param):
        # 1D params (biases and norm scales) never get weight decay
        if param.ndim <= 1:
            return True
        # Consolidate keywords; name.lower() handled once
        keywords = {
            "norm", "bn", "ln", "gn",  # Normalization
            "pos_embed", "position_embedding",  # Positional Info
            "rel_pos", "relative_position",  # Swin/ViT variants
            "cls_token", "mask_token", "dist_token",  # Special tokens
            "logit_scale"  # CLIP/Timm models
        }
        name = name.lower()
        return any(k in name for k in keywords)

    @staticmethod
    def get_layer_id(arch_type, name):
        map_dict = {
            'resnet50': {"conv1": 0, "bn1": 0, "layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4},
            'convnext': {"downsample_layers.0": 0, "stages.0": 1, "stages.1": 2, "stages.2": 3, "stages.3": 4},
            'swin': {"features.0": 0, "features.1": 1, "features.2": 1, "features.3": 2, "features.4": 2,
                     "features.5": 3, "features.6": 3, "features.7": 4}
        }
        mapping = map_dict[arch_type]
        return next((val for key, val in mapping.items() if key in name), 5)

    def build_llrd_groups(self, base_lr=1e-4, weight_decay=0.05, layer_decay=0.75):
        num_layers = 6
        lr_scales = {
            i: layer_decay ** (num_layers - 1 - i)
            for i in range(num_layers)
        }

        param_groups = {}
        for name, param in self.model.backbone.body.named_parameters():
            if not param.requires_grad:
                continue
            layer_id = self.get_layer_id(self.args.arch_type, name)
            decay_type = (
                "no_decay"
                if self.no_weight_decay(name, param)
                else "decay"
            )
            group_name = f"{layer_id}_{decay_type}"

            if group_name not in param_groups:
                param_groups[group_name] = {
                    "params": [],
                    "lr": base_lr * lr_scales[layer_id],
                    "weight_decay": (
                        0.0 if decay_type == "no_decay"
                        else weight_decay
                    ),
                }
            param_groups[group_name]["params"].append(param)
        return list(param_groups.values())

    def build_simple_param_groups(self, lr, weight_decay):
        decay_params = []
        no_decay_params = []

        for name, param in self.model.backbone.body.named_parameters():
            if not param.requires_grad:
                continue
            if self.no_weight_decay(name, param):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0}
        ]
        return param_groups

    def configure_training_strategy(self):
        strategy = self.args.strategy

        if strategy == "full":
            for p in self.model.backbone.parameters():
                p.requires_grad = True
        elif strategy == "frozen":
            for p in self.model.backbone.body.parameters():
                p.requires_grad = False
            for p in self.model.backbone.fpn.parameters():
                p.requires_grad = True
            for p in self.model.rpn.parameters():
                p.requires_grad = True
            for p in self.model.roi_heads.parameters():
                p.requires_grad = True
        elif strategy == "lora":
            pass  # since lora is activated at backbone level
        else:
            raise ValueError(strategy)

    def configure_optimizers(self):
        lr = self.args.lr
        wd = self.args.weight_decay
        lr_decay = self.args.lr_decay
        arch_type = self.args.arch_type

        if self.args.strategy == "full":
            param_groups = self.build_llrd_groups(
                base_lr=lr,
                weight_decay=wd,
                layer_decay=lr_decay
            )
        elif self.args.strategy in ["frozen", "lora"]:
            param_groups = self.build_simple_param_groups(lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown training strategy: {self.args.strategy.training_strategy}")

        optimizer = AdamW(param_groups)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps / 10)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=True,
                          persistent_workers=True, shuffle=True, drop_last=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=False, drop_last=False,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=False, drop_last=False,
                          collate_fn=collate_fn)

    def on_fit_start(self):
        self.train_start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(l for l in loss_dict.values())
        self.log_dict(loss_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
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

        self.log("test/bbox_mAP_50", bbox_results["map_50"])
        self.log("test/mask_mAP_50", segm_results["map_50"])

        self.dict_of_metrics['bbox_map_50'] = bbox_results["map_50"].detach().numpy()
        self.dict_of_metrics['mask_map_50'] = segm_results["map_50"].detach().numpy()

        self.test_map_bbox.reset()
        self.test_map_segm.reset()

        print("\n" + "=" * 30)
        print(f"FINAL TEST MASK mAP: {segm_results['map_50']:.4f}")
        print(f"FINAL TEST BBOX mAP: {bbox_results['map_50']:.4f}")
        print("=" * 30 + "\n")

        candidate_img = imread(paths['candidate'])
        candidate_img = F.to_tensor(candidate_img)

        with torch.no_grad():
            output_list = self.model(candidate_img.unsqueeze(0).to(self.device))
        output = output_list[0]
        save_path = os.path.join(paths['output'], "candidate_img.png")
        metric_path = os.path.join(paths['output'], "accuracy_metrics.yaml")
        write_dict_to_yaml(metric_path, self.dict_of_metrics)
        self.save_inference_image(candidate_img, output, save_path)

    def on_fit_end(self):
        efficiency_path = os.path.join(paths['output'], "efficiency_metrics.yaml")
        training_stats = {
            "train_time_sec": time.perf_counter() - self.train_start_time,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        }
        write_dict_to_yaml(efficiency_path, training_stats)

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


paths = {
    'resnet50': "/data_hdd/jazibmodels/RSFMCheckpoints/DeepForest_R50.pt",
    'swin': "/data_hdd/jazibmodels/RSFMCheckpoints/satlasnet_aerial_swin_v2_b_single_image.pth",
    'convnext': "/data_hdd/jazibmodels/RSFMCheckpoints/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    'data': "/data_hdd/jazibsdata/oam-tcd-coco-style-1024/",  # "/data_local_ssd/oam-tcd/"
    'output': "/data_hdd/jazibmodels/Fine_Tuning_Strategies/",
    'candidate': "/data_hdd/jazibsdata/oam-tcd-coco-style-1024/candidate_img/tile_93_1024_0.tif"
}

seed_dict = {'s1': 1234, 's2': 4319, 's3': 6147}


split_dict = {
    "full": {"train": [0, 1, 2, 3], "val": [4]},
    "half": {"train": [0, 1], "val": [4]},
    "quarter": {"train": [1], "val": [4]}
}


def main(args):
    L.seed_everything(seed_dict[args.seed])
    assert args.arch_type in ["resnet50", "swin", "convnext"], f"Unsupported arch_type: {args.arch_type}"
    if args.arch_type == "resnet50":
        backbone = ResNet50Backbone(
            checkpoint_path=paths['resnet50'] if args.use_pretrained else None,
            apply_lora=True if args.strategy == "lora" else False,
            lora_rank=args.lora_rank
        )
    elif args.arch_type == "convnext":
        backbone = ConvNeXtBackbone(
            backbone_type='small',
            checkpoint_path=paths['convnext'] if args.use_pretrained else None,
            apply_lora=True if args.strategy == "lora" else False,
            lora_rank=args.lora_rank
        )
    elif args.arch_type == "swin":
        backbone = TorchvisionSwinV2Backbone(
            checkpoint_path=paths['swin'] if args.use_pretrained else None,
            apply_lora=True if args.strategy == "lora" else False,
            lora_rank=args.lora_rank
        )
    else:
        print("Please choose the correct backbone.")
        sys.exit(1)

    backbone_with_fpn = BackboneWithFPN(body=backbone)

    transform = get_train_transforms()
    train_dataset = OAMTCDCOCODataset(root_dir=paths['data'],
                                      folds=split_dict[args.split]["train"],
                                      transforms=transform,
                                      return_masks=True)
    val_dataset = OAMTCDCOCODataset(root_dir=paths['data'],
                                    folds=split_dict[args.split]["val"],
                                    transforms=transform,
                                    return_masks=True)
    test_dataset = OAMTCDCOCODataset(root_dir=paths['data'],
                                     split='test',
                                     folds=None,
                                     return_masks=True)

    lightning_maskrcnn = LitMaskRCNN(
        backbone=backbone_with_fpn, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
        args=args
    )
    checkpoint_callback = ModelCheckpoint(dirpath=paths['output'],
                                          # every_n_epochs=int(args.max_epochs / 3),
                                          save_last=True)
    logger = TensorBoardLogger(save_dir=paths['output'],
                               name="",
                               default_hp_metric=False)

    early_stop_callback = EarlyStopping(
        monitor="val/bbox_mAP_50",
        patience=3,
        verbose=True,
        mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callback_list = [checkpoint_callback, early_stop_callback, lr_monitor]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=callback_list,
        logger=logger,
        precision="16-mixed",
        check_val_every_n_epoch=5,
        # limit_train_batches=100,
        # limit_val_batches=2,
        # limit_test_batches=2,
        log_every_n_steps=100
    )

    trainer.fit(model=lightning_maskrcnn)
    trainer.test()


if __name__ == '__main__':
    args = get_args()
    # create the output_directory and update path
    experiment_name = f"{args.seed}_{args.arch_type}_{args.strategy}_{args.split}"
    paths['output'] = os.path.join(paths['output'], experiment_name)
    os.makedirs(paths['output'], exist_ok=True)
    # write args to yaml for record keeping
    args_yml_fp = os.path.join(paths['output'], "args.yaml")
    write_dict_to_yaml(args_yml_fp, args.__dict__)
    # run the training
    main(args)
