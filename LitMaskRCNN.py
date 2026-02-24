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
from dataset import OAMTCDCOCODataset
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


L.seed_everything(1234)  # 7890


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
    parser.add_argument("--data_path", type=str,
                        default="")
    parser.add_argument("--output_dir", type=str,
                        default="")
    parser.add_argument("--candidate_path", type=str,
                        default="")

    parser.add_argument("--train_folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--val_folds", type=int, nargs="+", default=[0])

    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--arch_type", type=str, default="")

    return parser.parse_args()


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def collate_fn(batch):
    return tuple(zip(*batch))


class GradualUnfreezing(BaseFinetuning):
    def __init__(self, backbone_type: str, base_lr: float = 1e-4, stages=(5, 10, 15)):
        super().__init__()
        self.backbone_type = backbone_type
        self.base_lr = base_lr
        self.stages = stages

    def freeze_before_training(self, pl_module):
        # Freeze backbone completely
        self.freeze(pl_module.backbone)
        # Make sure RPN + ROI heads train
        self.make_trainable(pl_module.rpn)
        self.make_trainable(pl_module.roi_heads)

    def finetune_function(self, pl_module, epoch, optimizer):
        if epoch == self.stages[0]:
            if self.backbone_type == "resnet":
                self._unfreeze_and_add(pl_module.backbone.layer4, optimizer)
            elif self.backbone_type == "swin":
                self._unfreeze_and_add(pl_module.backbone.features[6], optimizer)
                self._unfreeze_and_add(pl_module.backbone.features[7], optimizer)
            elif self.backbone_type == "vit":
                for i in range(8, 12):
                    self._unfreeze_and_add(pl_module.backbone.blocks[i], optimizer)

        if epoch == self.stages[1]:
            if self.backbone_type == "resnet":
                self._unfreeze_and_add(pl_module.backbone.layer3, optimizer)
            elif self.backbone_type == "swin":
                self._unfreeze_and_add(pl_module.backbone.features[4], optimizer)
                self._unfreeze_and_add(pl_module.backbone.features[5], optimizer)
            elif self.backbone_type == "vit":
                for i in range(4, 8):
                    self._unfreeze_and_add(pl_module.backbone.blocks[i], optimizer)

        if epoch == self.stages[2]:
            self.unfreeze_and_add_param_group(pl_module.backbone, optimizer)

    def _unfreeze_and_add(self, module, optimizer):
        self.unfreeze_and_add_param_group(
            module,
            optimizer,
            lr=self.base_lr  # * 0.1  # backbone LR lower than head
        )


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
        arch = self.args.arch_type
        if arch == 'vit':
            return self.model.backbone.vit.vit.model
        elif arch == 'swin':
            return self.model.backbone.body.body
        else:  # resnet
            return self.model.backbone.body

    @staticmethod
    def build_mask_rcnn(backbone, num_classes, img_size, fast=False):
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        )
        model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=img_size,
            max_size=img_size
        )
        # 1. Remove low-confidence detections
        model.roi_heads.score_thresh = 0.5  # try 0.3–0.7
        # 2. Stronger Non-Max Suppression
        model.roi_heads.nms_thresh = 0.3  # default ~0.5
        # 3. Hard cap detections per image
        model.roi_heads.detections_per_img = 100
        if fast:
            model.rpn_pre_nms_top_n_train = 1000
            model.rpn_post_nms_top_n_train = 300
            model.rpn_batch_size_per_image = 128
        return model

    def split_decay(self, params):
        decay = []
        no_decay = []
        for n, p in self.named_parameters():
            if id(p) in params:
                if p.ndim < 2 or "bias" in n or "norm" in n or "pos_embed" in n:
                    no_decay.append(p)
                else:
                    decay.append(p)
        return decay, no_decay

    def get_tier_mapper(self):
        """Maps backbone into thirds based on architecture type."""
        arch = self.args.arch_type
        # m = self.backbone

        if "vit" == arch:
            m = self.model.backbone.vit.vit.model
            # ViT-Base: Tier 1 (0-3), Tier 2 (4-7), Tier 3 (8-11)
            # Tier 1: patch_embed + blocks 0-3
            tier1 = (list(m.patch_embed.parameters()) + list(m.blocks[:4].parameters()))
            for p in tier1: p.requires_grad = True
            # Tier 2: blocks 4-7
            tier2 = list(m.blocks[4:8].parameters())
            for p in tier2: p.requires_grad = True
            # Tier 3: blocks 8-11
            tier3 = list(m.blocks[8:].parameters())
            for p in tier3: p.requires_grad = True
            return tier1, tier2, tier3
        elif "swin" == arch:
            m = self.model.backbone.body.body
            # m.features[0] is PatchEmbed
            # Tier 1: Stages 1 & 2 (indices 1 & 3) + PatchEmbed & downsampling (index 2)
            tier1 = list(m.features[0].parameters()) + list(m.features[1].parameters()) + \
                    list(m.features[2].parameters()) + list(m.features[3].parameters())
            for p in tier1: p.requires_grad = True
            # Tier 2: Stage 3 (index 5) + downsampling (index 4)
            tier2 = list(m.features[4].parameters()) + list(m.features[5].parameters())
            for p in tier2: p.requires_grad = True
            # Tier 3: Stage 4 (index 7) + downsampling (index 6)
            tier3 = list(m.features[6].parameters()) + list(m.features[7].parameters())
            for p in tier3: p.requires_grad = True
            return tier1, tier2, tier3
        elif "resnet" == arch:
            m = self.model.backbone.body
            # Tier 1: Layers 1 & 2 + initial stems
            tier1 = list(m.conv1.parameters()) + list(m.bn1.parameters()) + list(m.layer1.parameters()) + \
                    list(m.layer2.parameters())
            for p in tier1: p.requires_grad = True
            # Tier 2: Layer 3
            tier2 = list(m.layer3.parameters())
            for p in tier2: p.requires_grad = True
            # Tier 3: Layer 4
            tier3 = list(m.layer4.parameters())
            for p in tier3: p.requires_grad = True
            return tier1, tier2, tier3

    def configure_optimizers(self):
        lr = self.args.lr
        wd = self.args.weight_decay

        if self.args.arch_type == 'vit':
            body_params = set(id(p) for p in self.model.backbone.vit.vit.parameters())
        else:
            body_params = set(id(p) for p in self.model.backbone.body.parameters())
        # Remember Neck/FPN params do not change and are part of head params
        head_params = [p for p in self.model.parameters() if id(p) not in body_params and p.requires_grad]

        if self.args.strategy == 'adaptive':
            tier_1, tier_2, tier_3 = self.get_tier_mapper()
            param_groups = [
                {'params': tier_1, 'lr': lr * 0.125, 'weight_decay': wd},
                {'params': tier_2, 'lr': lr * 0.25, 'weight_decay': wd},
                {'params': tier_3, 'lr': lr * 0.50, 'weight_decay': wd},
                {'params': head_params, 'lr': lr, 'weight_decay': wd}
            ]
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
        elif self.args.strategy in ['frozen', 'full']:
            decay_params = []
            no_decay_params = []
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim < 2 or "bias" in n or "norm" in n or "pos_embed" in n:
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
                param_groups = [
                    {'params': decay_params, 'weight_decay': wd},
                    {'params': no_decay_params, 'weight_decay': 0.0}
                ]
                optimizer = torch.optim.AdamW(param_groups, lr=lr)
        else:  # gradual unfreezing
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=lr,
                weight_decay=wd
            )

        total_steps = self.trainer.estimated_stepping_batches
        # warmup_steps = self.args.warmup_steps
        warmup_steps = int(total_steps/10)

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


def main(args):
    # initializing the model
    assert args.arch_type in {"vit", "swin", "resnet"}, \
        f"Unsupported arch_type: {args.arch_type}"
    if args.arch_type == 'vit':
        backbone_with_fpn = ViTDetBackbone(
            img_size=args.img_size,
            out_channels=128 if args.fast else 256
        )
        if args.use_pretrained:
            backbone_with_fpn.vit.load_checkpoint(args.ckpt_path)
        if args.use_lora:
            backbone_with_fpn.vit.apply_lora(args.lora_rank)
            # args.freeze_backbone = False  # this is automatically done in .apply_lora and is not needed
        # if args.freeze_backbone:
        #     backbone_with_fpn.vit.freeze_parameters()
    elif args.arch_type == 'swin':
        if not args.use_lora:
            swin = AltTorchvisionSwinV2Backbone(
                checkpoint_path=args.ckpt_path if args.use_pretrained else None
            )
        else:
            swin = TorchvisionSwinV2Backbone(
                checkpoint_path=args.ckpt_path if args.use_pretrained else None,
                apply_lora=args.use_lora,
                lora_rank=args.lora_rank
            )
        backbone_with_fpn = BackboneWithFPN(body=swin,
                                            out_channels=128 if args.fast else 256)
    elif args.arch_type == 'resnet':
        if not args.use_lora:
            resnet50 = AltTorchvisionResNet50Backbone(
                checkpoint_path=args.ckpt_path if args.use_pretrained else None
            )
        else:
            resnet50 = ResNet50Backbone(
                checkpoint_path=args.ckpt_path if args.use_pretrained else None,
                apply_lora=args.use_lora,
                lora_rank=args.lora_rank
            )
        backbone_with_fpn = BackboneWithFPN(body=resnet50)
    else:
        print("Please choose the correct backbone.")
        sys.exit(1)

    train_dataset = OAMTCDCOCODataset(root_dir=args.data_path,
                                      folds=args.train_folds,
                                      return_masks=True)
    val_dataset = OAMTCDCOCODataset(root_dir=args.data_path,
                                    folds=args.val_folds,
                                    return_masks=True)
    test_dataset = OAMTCDCOCODataset(root_dir=args.data_path,
                                     split='test',
                                     folds=None,
                                     return_masks=True)
    lightning_maskrcnn = LitMaskRCNN(
        backbone=backbone_with_fpn, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
        args=args
    )

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          # every_n_epochs=int(args.max_epochs / 3),
                                          save_last=True)
    logger = TensorBoardLogger(save_dir=args.output_dir,
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

    if args.strategy == 'gradual':
        finetune_cb = GradualUnfreezing(backbone_type=args.arch_type, base_lr=args.lr)
        callback_list.append(finetune_cb)

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
        log_every_n_steps=50
    )

    trainer.fit(model=lightning_maskrcnn)
    trainer.test()


if __name__ == '__main__':
    # args = DotDict(
    #     num_classes=3,
    #     num_workers=8,
    #     batch_size=1,
    #     lr=1e-4,
    #     weight_decay=0.001,
    #     warmup_steps=10,
    #     use_lora=True,
    #     lora_rank=8,
    #     use_pretrained=True,
    #     ckpt_path= "/home/jazib/projects/RSFMCheckpoints/satlasnet_aerial_swin_v2_b_single_image.pth",
    #     # swin: "/home/jazib/projects/RSFMCheckpoints/satlasnet_aerial_swin_v2_b_single_image.pth"
    #     # r50: "/home/jazib/projects/RSFMCheckpoints/DeepForest_R50.pt"
    #     # vit: "/home/jazib/projects/savedmodels/meta_vitbase16_bench.pth"
    #     freeze_backbone=True,
    #     data_path="/home/jazib/projects/data/oam-tcd-coco-style-1024/",
    #     output_dir="./experiments/exp_gradual/",
    #     candidate_path="/home/jazib/projects/data/oam-tcd-coco-style-1024/candidate_img/tile_93_1024_0.tif",
    #     train_folds=[0],  # [0, 1, 2, 3]
    #     val_folds=[4],
    #     max_epochs=20,
    #     img_size=1024,
    #     strategy='lora',
    #     arch_type="swin"
    # )
    args = get_args()
    main(args)
