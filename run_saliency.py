##
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tifffile import imread
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_png
from run_collect_preds import collate_fn, DotDict, init_backbone, init_datasets, load_lit_model
from run_collect_preds import get_test_loader, mask_to_polygons
import yaml
import torchvision
from torchvision.transforms.v2 import ToTensor


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
    category_colors = {1: "red", 2: "yellow"}

    # Colors corresponding to each bounding box
    colors = [
        category_colors.get(int(label), "green")
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


SPLIT_DICT = {
    "full": {"train": [0, 1, 2, 3], "val": [4]},
    "half": {"train": [0, 1], "val": [4]},
    "quarter": {"train": [1], "val": [4]}
}


MODEL_ROOT = "/mnt/cluster/data_hdd/jazibmodels/Fine_Tuning_Strategies/"
DATA_ROOT = "/mnt/cluster/data_hdd/jazibsdata/oam-tcd-coco-style-1024/"
ex_model = 's1_convnext_full_full'

# loading gt annotations
anno_source = "/home/jazib/projects/data/oam-tcd-coco-style-1024/coco_annotations_test.json"
with open(anno_source, "r") as fr:
    annos = json.load(fr)
# loading the example image
example_root = "/home/jazib/projects/data/oam-tcd-3-test-imgs/"
example_file_name = "tile_238_1024_1024.tif"
example_img_path = os.path.join(example_root, example_file_name)
ex_img = imread(example_img_path)

# get the gt annos and plot them on the original image
gt_annos = get_annotations_for_file(annos, example_file_name)
plot_annotations(example_img_path, gt_annos)

# load the pretrained model
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


# get the outputs
input = ToTensor()(ex_img).unsqueeze(0)
lit_model.to("cuda")
lit_model.model.eval()
with torch.no_grad():
    output = lit_model.model(input.to('cuda'))

# plot the predicted bboxes
plot_predictions_bbox(input.squeeze(0), output[0], threshold=0.5)
##
# here starts the saliency stuff
# https://xaitk-saliency.readthedocs.io/en/latest/examples/DRISE.html

# reference preds, first two
ind_1, ind_2 = 4, 16
ref_bboxes = []
ref_scores = []

ref_bboxes.append(output[0]['boxes'][ind_1])
ref_bboxes.append(output[0]['boxes'][ind_2])

ref_scores.append(output[0]['scores'][ind_1])
ref_scores.append(output[0]['scores'][ind_2])

ref_bb = []
for bb in ref_bboxes:
    ref_bb.append(bb.detach().cpu().tolist())

ref_sc = []
for sc in ref_scores:
    ref_sc.append(sc.detach().cpu().item())

##
# since I have the reference bb and sc, let's plot them on top of the original image
_, axs = plt.subplots(1, 2, figsize=(10,4))
for i, bbox in enumerate(ref_bb):
    axs[i].imshow(ex_img)
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=1,
        edgecolor='r',
        facecolor="none"
    )
    axs[i].add_patch(rect)
    axs[i].set_title(f"detection #{i+1}")
    axs[i].axis("off")

plt.show()
##
# define saliency alg - Drise for me

from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack

sal_generator = DRISEStack(n=200, s=8, p1=0.5, seed=0, threads=4)

model_mean = [0.485, 0.456, 0.406]
blackbox_fill = np.uint8(np.asarray(model_mean) * 255)
sal_generator.fill = blackbox_fill

sal_maps = sal_generator.generate(
    ex_img,
    np.array(ref_bb),
    np.array(ref_sc),
    lit_model.model,
)
