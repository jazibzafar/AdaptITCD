##
import torch
import os
from modules import ViTDetBackbone, TorchvisionSwinV2Backbone, ResNet50Backbone, BackboneWithFPN, ConvNeXtBackbone
from torch.utils.data import DataLoader
import cv2
import numpy as np
from dataset import OAMTCDCOCODataset
from SimplifiedMaskRCNN import LitMaskRCNN
import yaml
import json
from tqdm.auto import tqdm
import sys



def collate_fn(batch):
    return tuple(zip(*batch))


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def init_backbone(args):
    if args.arch_type == "resnet50":
        backbone = ResNet50Backbone(
            checkpoint_path=None,
            apply_lora=True if args.strategy == "lora" else False,
            lora_rank=args.lora_rank
        )
    elif args.arch_type == "convnext":
        backbone = ConvNeXtBackbone(
            backbone_type='small',
            checkpoint_path=None,
            apply_lora=True if args.strategy == "lora" else False,
            lora_rank=args.lora_rank
        )
    elif args.arch_type == "swin":
        backbone = TorchvisionSwinV2Backbone(
            checkpoint_path=None,
            apply_lora=True if args.strategy == "lora" else False,
            lora_rank=args.lora_rank
        )
    else:
        print("Please choose the correct backbone.")
        sys.exit(1)

    backbone_with_fpn = BackboneWithFPN(body=backbone)
    return backbone_with_fpn


def init_datasets(data_root, split_dict, args):
    train_dataset = OAMTCDCOCODataset(
        root_dir=data_root,
        folds=split_dict[args.split]["train"],
        transforms=None,
        return_masks=True
    )
    val_dataset = OAMTCDCOCODataset(
        root_dir=data_root,
        folds=split_dict[args.split]["val"],
        transforms=None,
        return_masks=True)
    test_dataset = OAMTCDCOCODataset(
        root_dir=data_root,
        split='test',
        folds=None,
        return_masks=True
    )
    return train_dataset, val_dataset, test_dataset


def load_lit_model(ckpt_path, backbone_with_fpn, train_dataset, val_dataset, test_dataset, args):
    return LitMaskRCNN.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        backbone=backbone_with_fpn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        args=args
    )


def get_test_loader(test_dataset, args):
    return DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )


def mask_to_polygons(mask):
    """Convert a binary mask to polygon segmentation."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = []
    for contour in contours:
        # Polygon needs at least 3 points
        if contour.shape[0] < 3:
            continue
        polygon = contour.reshape(-1, 2).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def predict(prediction_model, dataloader, score_threshold=0.0, device="cuda"):
    global_anno_counter = 0
    anno_output_list = []
    pbar = tqdm(dataloader, desc="Predicting")

    prediction_model.model.eval()

    with torch.no_grad():
        for batch in pbar:
            images, targets = batch
            images = [image.to(device) for image in images]
            outputs = prediction_model.model(images)

            # Iterate over images in batch
            for target, output in zip(targets, outputs):
                # Image ID
                image_id = target["image_id"]
                if torch.is_tensor(image_id):
                    image_id = image_id.item()

                # Predictions for this image
                boxes = output["boxes"]
                labels = output["labels"]
                scores = output["scores"]
                masks = output["masks"]
                for box, label, score, mask in zip(boxes, labels, scores, masks):
                    # Score filtering
                    score = score.detach().cpu().item()
                    if score < score_threshold:
                        continue

                    # Box
                    box = box.detach().cpu().numpy()

                    # Label
                    label = label.detach().cpu().item()

                    # Mask
                    # torchvision Mask R-CNN:
                    # mask shape = [1, H, W]
                    mask = mask.detach().cpu().numpy()
                    mask = mask[0]
                    mask_binary = (mask >= 0.5).astype(np.uint8)
                    segmentation = mask_to_polygons(mask_binary)
                    if len(segmentation) == 0:
                        continue
                    # Area
                    area = float(mask_binary.sum())
                    # Store annotation
                    anno_output_list.append({
                        "anno_id": global_anno_counter,
                        "image_id": image_id,
                        "segmentation": segmentation,
                        "bbox": box.tolist(),
                        "category_id": label,
                        "area": area,
                        "score": score,
                    })
                    global_anno_counter += 1
            pbar.set_postfix(annotations=len(anno_output_list))
    return anno_output_list


MODEL_ROOT = "/data_hdd/jazibmodels/Fine_Tuning_Strategies/"
DATA_ROOT = "/data_hdd/jazibsdata/oam-tcd-coco-style-1024/"
SAVE_ROOT = "/data_hdd/jazibsdata/m2_uncertainty_quant/"

SWIN_LIST = [
    's1_swin_full_full', 's1_swin_lora_full', 's1_swin_frozen_quarter', 's1_swin_frozen_full', 's1_swin_full_quarter',
    's1_swin_lora_quarter', 's2_swin_full_full', 's2_swin_full_quarter', 's2_swin_frozen_full',
    's2_swin_frozen_quarter', 's2_swin_lora_full', 's2_swin_lora_quarter', 's3_swin_frozen_quarter',
    's3_swin_full_full', 's3_swin_full_quarter', 's3_swin_lora_quarter', 's3_swin_lora_full', 's3_swin_frozen_full'
]

CONV_LIST = [
    's1_convnext_full_full', 's1_convnext_lora_full', 's1_convnext_frozen_full', 's1_convnext_frozen_quarter',
    's1_convnext_lora_quarter', 's1_convnext_full_quarter', 's2_convnext_lora_quarter', 's2_convnext_full_quarter',
    's2_convnext_frozen_quarter', 's2_convnext_frozen_full', 's2_convnext_lora_full', 's3_convnext_lora_quarter',
    's3_convnext_full_quarter', 's2_convnext_full_full', 's3_convnext_lora_full', 's3_convnext_frozen_quarter',
    's3_convnext_frozen_full', 's3_convnext_full_full'
]

R50_LIST = [
    's1_resnet50_full_full', 's1_resnet50_frozen_quarter', 's1_resnet50_frozen_full', 's1_resnet50_full_quarter',
    's2_resnet50_frozen_full', 's2_resnet50_full_full', 's2_resnet50_frozen_quarter', 's2_resnet50_full_quarter',
    's3_resnet50_frozen_full', 's3_resnet50_frozen_quarter', 's3_resnet50_full_full', 's3_resnet50_full_quarter'
]


SPLIT_DICT = {
    "full": {"train": [0, 1, 2, 3], "val": [4]},
    "half": {"train": [0, 1], "val": [4]},
    "quarter": {"train": [1], "val": [4]}
}


def main():
    for saved_model in SWIN_LIST:
        current_ckpt_path = os.path.join(MODEL_ROOT, saved_model, "last.ckpt")
        with open(os.path.join(MODEL_ROOT, current_ckpt_path, "args.yaml"), 'r') as stream:
            args = yaml.safe_load(stream)
        args = DotDict(args)

        print(f"starting collecting annotations for {saved_model}\n")
        backbone = init_backbone(args)
        train_d, val_d, test_d = init_datasets(DATA_ROOT, SPLIT_DICT, args)
        lit_model = load_lit_model(
            ckpt_path=current_ckpt_path,
            backbone_with_fpn=backbone,
            train_dataset=train_d,
            val_dataset=val_d,
            test_dataset=test_d,
            args=args
        )
        loader = get_test_loader(test_d, args)
        predicted_annos_list = predict(
            prediction_model=lit_model,
            dataloader=loader,
            score_threshold=0.01
        )
        print(f"the collected annotations contain: {len(predicted_annos_list)} annos.\n")
        save_file = os.path.join(SAVE_ROOT, f"{saved_model}_predict.json")
        with open(save_file, "w") as ji:
            json.dump(predicted_annos_list, ji)
        print(f"annotations saved at {save_file}\n")


if __name__ == '__main__':
    main()