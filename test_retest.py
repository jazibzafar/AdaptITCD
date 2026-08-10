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


def collate_fn(batch):
    return tuple(zip(*batch))


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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


# TODO: Get a list of all the checkpoints
model_root = "/home/jazib/projects/ft_uq_bbs/"
data_root = "/home/jazib/projects/data/oam-tcd-coco-style-1024/"
list_ckpt = [
    's1_convnext_full_full',
    's1_convnext_lora_full'
]
current_ckpt = list_ckpt[0]



# TODO: Path augmentation for each checkpoint os.path.join(root, ckpt[i])
ckpt_path = os.path.join(model_root, current_ckpt, "last.ckpt")

with open(os.path.join(model_root, current_ckpt, "args.yaml"), 'r') as stream:
    args = yaml.safe_load(stream)

args = DotDict(args)


##

# [X] seed, backbone, strat, split separation
seed, backbone, strategy, split = current_ckpt.split("_")


split_dict = {
    "full": {"train": [0, 1, 2, 3], "val": [4]},
    "half": {"train": [0, 1], "val": [4]},
    "quarter": {"train": [1], "val": [4]}
}


backbone = ConvNeXtBackbone(
            backbone_type='small',
            checkpoint_path=None,
            apply_lora=True if args.strategy == "lora" else False,
            lora_rank=args.lora_rank
        )
backbone_with_fpn = BackboneWithFPN(body=backbone)


# DONE: Load the dataset
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

# TODO: Load the lightning model
model = LitMaskRCNN.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    backbone=backbone_with_fpn,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    args=args
)

# DONE: Dataloader
retest_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_fn
)


# DONE: Define the predict step
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


attempt_anno_output_list = predict(
    prediction_model=model,
    dataloader=retest_dataloader,
    score_threshold=0.01
)

##
save_file = f"./statistics/{current_ckpt}_predict.json"

with open(save_file, "w") as ji:
    json.dump(attempt_anno_output_list, ji)

##
# with open(save_file, "r") as jr:
#     at3 = json.load(jr)

##
# DONE: Sensible output test [PASSED]
# with torch.no_grad():
#     model.eval()
#     dddd = model.model(torch.rand(1, 3, 1024, 1024).to("cuda"))

# DONE: Actual output test [PASSED]
# import matplotlib.pyplot as plt
#
#
# sample = next(iter(retest_dataloader))
# img, label = sample
#
# with torch.no_grad():
#     model.eval()
#     out = model.model(img.unsqueeze(0).to("cuda"))
#
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(img.permute(1,2,0).numpy())
# ax[0].axis('off')
# ax[1].imshow(label['masks'].squeeze(0).numpy())
# ax[1].axis('off')
# plt.show()

# from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
#
# def show_out(img_tensor, pred):
#     img_uint8 = (img_tensor.cpu().clamp(0, 1) * 255).byte()
#     boxes = pred["boxes"]
#     masks = pred["masks"].squeeze(1) > 0.5 if "masks" in pred else None
#     result_img = img_uint8
#     if boxes.numel() > 0:
#         result_img = draw_bounding_boxes(result_img, boxes, colors="red", width=3)
#         if masks is not None and masks.any():
#             result_img = draw_segmentation_masks(result_img, masks, alpha=0.5, colors="blue")
#     plt.imshow(result_img.permute(1,2,0))
#     plt.axis("off")
#     plt.show()
#
#
# show_out(img, out[0])
