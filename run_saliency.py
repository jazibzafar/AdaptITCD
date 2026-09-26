##
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tifffile import imread
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_annotations_for_file(coco_data, file_name):
    # Find image ID corresponding to file_name
    image_id = next(
        image["id"]
        for image in coco_data["images"]
        if image["file_name"] == file_name
    )
    # Get all annotations with this image_id
    annotations = [
        ann
        for ann in coco_data["annotations"]
        if ann["image_id"] == image_id
    ]
    return annotations


def plot_annotations(image_path, annotations):
    image = imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)
    # Two categories
    category_colors = {1: "red", 2: "blue"}
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        category_id = annotation["category_id"]
        color = category_colors.get(category_id, "yellow")

        rectangle = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rectangle)

    ax.axis("off")
    plt.tight_layout()
    plt.show()


anno_source = "/home/jazib/projects/data/oam-tcd-coco-style-1024/coco_annotations_test.json"
with open(anno_source, "r") as fr:
    annos = json.load(fr)

example_root = "/home/jazib/projects/data/oam-tcd-3-test-imgs/"
example_file_name = "tile_238_1024_1024.tif"
example_img_path = os.path.join(example_root, example_file_name)
ex_img = imread(example_img_path)
# {'id': 1065,
#  'oam_image_id': 238,
#  'biome_name': 'Sarmatic mixed forests',
#  'biome_id': 4,
#  'file_name': 'tile_238_1024_1024.tif',
#  'width': 1024,
#  'height': 1024}

gt_annos = get_annotations_for_file(annos, example_file_name)
plot_annotations(example_img_path, gt_annos)
##
from run_collect_preds import collate_fn, DotDict, init_backbone, init_datasets, load_lit_model
from run_collect_preds import get_test_loader, mask_to_polygons
import yaml
import torchvision
from torchvision.transforms.v2 import ToTensor


input = ToTensor()(ex_img).unsqueeze(0)

##
SPLIT_DICT = {
    "full": {"train": [0, 1, 2, 3], "val": [4]},
    "half": {"train": [0, 1], "val": [4]},
    "quarter": {"train": [1], "val": [4]}
}


MODEL_ROOT = "/mnt/cluster/data_hdd/jazibmodels/Fine_Tuning_Strategies/"
DATA_ROOT = "/mnt/cluster/data_hdd/jazibsdata/oam-tcd-coco-style-1024/"
ex_model = 's1_convnext_full_full'

current_ckpt_path = os.path.join(MODEL_ROOT, ex_model, "last.ckpt")
with open(os.path.join(MODEL_ROOT, ex_model, "args.yaml"), 'r') as stream:
    args = yaml.safe_load(stream)
args = DotDict(args)

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

##
lit_model.to("cuda")
lit_model.model.eval()
with torch.no_grad():
    output = lit_model.model(input.to('cuda'))
##
from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_png


def plot_predictions_bbox(img_tensor, pred, threshold=0.5,):
    # Convert image to uint8
    img_uint8 = (img_tensor.cpu().clamp(0, 1) * 255).byte()
    # Filter predictions
    scores = pred["scores"].cpu()
    labels = pred["labels"].cpu()
    boxes = pred["boxes"].cpu()

    keep = scores > threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    # Category colors
    category_colors = {1: "red", 2: "blue"}

    # Colors corresponding to each bounding box
    colors = [
        category_colors.get(int(label), "yellow")
        for label in labels
    ]

    # Text displayed on each box
    box_labels = [
        f"cat {int(label)}: {score:.2f}"
        for label, score in zip(labels, scores)
    ]
    # Draw boxes
    if boxes.numel() > 0:
        result_img = draw_bounding_boxes(
            img_uint8,
            boxes,
            labels=box_labels,
            colors=colors,
            width=3,
            font_size=16,
        )
    else:
        result_img = img_uint8
    # Convert CHW -> HWC for matplotlib
    result_img = result_img.permute(1, 2, 0)
    # Display
    plt.figure(figsize=(12, 12))
    plt.imshow(result_img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


plot_predictions_bbox(input.squeeze(0), output[0], threshold=0.4)
##
