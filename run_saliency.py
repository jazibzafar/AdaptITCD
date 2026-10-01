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
ex_model = 's1_resnet50_full_full'

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
ind_1, ind_2 = 12, 19
ref_bboxes = []
ref_scores = []
ref_labels = []

ref_bboxes.append(output[0]['boxes'][ind_1])
ref_bboxes.append(output[0]['boxes'][ind_2])

ref_scores.append(output[0]['scores'][ind_1])
ref_scores.append(output[0]['scores'][ind_2])

ref_labels.append(output[0]['labels'][ind_1])
ref_labels.append(output[0]['labels'][ind_2])

ref_bb = []
for bb in ref_bboxes:
    ref_bb.append(bb.detach().cpu().tolist())

ref_sc = []
for sc in ref_scores:
    ref_sc.append(sc.detach().cpu().item())

ref_lb = []
for lb in ref_labels:
    ref_lb.append(lb.detach().cpu().item())

# num_references x num_labels (i.e. 2 x 2)
class_scores = np.zeros((2, 2), dtype=np.float32)
for i, lb in enumerate(ref_labels):
    if lb == 1:
        class_scores[i, 0] = 1.0
    else:
        class_scores[i, 1] = 1.0



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


from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_image_io.bbox import AxisAlignedBoundingBox


class MaskRCNNBlackBox(DetectImageObjects):

    def __init__(
        self,
        model,
        device="cuda",
        batch_size=4,
        score_threshold=0.0,
    ):
        self.model = model
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.score_threshold = score_threshold

        self.model.to(self.device)
        self.model.eval()

    def get_config(self):
        """
        Required by the abstract DetectImageObjects interface.

        The model itself is intentionally not serialized here because
        we're passing an already-loaded PyTorch model.
        """
        return {
            "device": str(self.device),
            "batch_size": self.batch_size,
            "score_threshold": self.score_threshold,
        }

    def detect_objects(self, images):

        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()

        images = np.asarray(images)

        # [H, W, C] -> [1, H, W, C]
        if images.ndim == 3:
            images = images[None, ...]

        if images.ndim != 4:
            raise ValueError(
                f"Expected [N,H,W,C] or [H,W,C], got {images.shape}"
            )

        n_images = images.shape[0]

        all_detections = []

        for start in range(0, n_images, self.batch_size):

            end = min(start + self.batch_size, n_images)

            batch_np = images[start:end]

            # HWC -> CHW
            batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2)

            # uint8 [0,255] -> float [0,1]
            batch = batch.float() / 255.0

            # torchvision detection models expect:
            # list of [C,H,W] tensors
            batch_list = [
                img.to(self.device)
                for img in batch
            ]

            with torch.no_grad():
                predictions = self.model(batch_list)

            for pred in predictions:

                boxes = pred["boxes"].detach().cpu()
                labels = pred["labels"].detach().cpu()
                scores = pred["scores"].detach().cpu()

                image_detections = []

                for box, label, score in zip(
                    boxes,
                    labels,
                    scores,
                ):

                    score = float(score)
                    label = int(label)

                    if score < self.score_threshold:
                        continue

                    x1, y1, x2, y2 = box.tolist()

                    bbox = AxisAlignedBoundingBox(
                        min_vertex=np.array([x1, y1]),
                        max_vertex=np.array([x2, y2]),
                    )

                    score_dict = {
                        1: 0.0,
                        2: 0.0,
                    }

                    if label not in score_dict:
                        raise ValueError(
                            f"Unexpected Mask R-CNN class label: {label}"
                        )

                    score_dict[label] = score

                    image_detections.append(
                        (bbox, score_dict)
                    )

                all_detections.append(image_detections)

        return all_detections


blackbox_model = MaskRCNNBlackBox(
    model=lit_model.model,
    device="cuda",
    batch_size=1,
    score_threshold=0.0,
)

sal_generator = DRISEStack(n=200, s=8, p1=0.5, seed=0, threads=4)

model_mean = [0.5, 0.5, 0.5]
blackbox_fill = np.uint8(np.asarray(model_mean) * 255)
sal_generator.fill = blackbox_fill

sal_maps = sal_generator.generate(
    ex_img,
    np.array(ref_bb),
    class_scores,
    blackbox_model,
    objectness=np.array(ref_sc)
)

##
# bbox 1
colorbar_kwargs = {
    "fraction": 0.046 * (ex_img.shape[0] / ex_img.shape[1]),
    "pad": 0.04,
}
idx = 0
bbox = ref_bb[idx]
plt.figure(figsize=(12, 8))
plt.axis("off")
plt.imshow(ex_img, alpha=0.7)
plt.clim(-1, 1)
plt.imshow(sal_maps[idx], cmap="jet", alpha=0.3)
ax = plt.gca()
rect = patches.Rectangle(
    (bbox[0], bbox[1]),
    bbox[2] - bbox[0],
    bbox[3] - bbox[1],
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)
_ = plt.colorbar(**colorbar_kwargs)
plt.show()

##
from utils_saliency import classify_tree_predictions


classified = classify_tree_predictions(gt_annos, output[0], tau=0.5)
##
from utils_saliency import process_saliency_map


processed = process_saliency_map(sal_maps[0], kappa=0.85)

##

colorbar_kwargs = {
    "fraction": 0.046 * (ex_img.shape[0] / ex_img.shape[1]),
    "pad": 0.04,
}
idx = 0
bbox = ref_bb[idx]
plt.figure(figsize=(12, 8))
plt.axis("off")
plt.imshow(ex_img, alpha=0.7)
plt.clim(-1, 1)
plt.imshow(processed["thresholded_map"], cmap="jet", alpha=0.3)
ax = plt.gca()
rect = patches.Rectangle(
    (bbox[0], bbox[1]),
    bbox[2] - bbox[0],
    bbox[3] - bbox[1],
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)
_ = plt.colorbar(**colorbar_kwargs)
plt.show()



##
def quantify_saliency(
    saliency_map,
    ground_truth_mask,
    prediction_mask,
):
    """
    Quantify where saliency is located relative to ground truth
    and prediction masks.

    Parameters
    ----------
    saliency_map : np.ndarray
        2D normalized saliency map.

        Ideally this is the single-component normalized saliency
        map produced by process_saliency_map().

    ground_truth_mask : np.ndarray
        Binary 2D ground-truth mask.

    prediction_mask : np.ndarray
        Binary 2D prediction mask.

    Returns
    -------
    dict
        Contains:

        saliency_inside_gt
            Fraction of total saliency inside GT.

        saliency_outside_gt
            Fraction of total saliency outside GT.

        saliency_inside_prediction
            Fraction of total saliency inside prediction.

        saliency_overlap_gt_prediction
            Fraction of total saliency inside both GT and prediction.

        raw_saliency_inside_gt
            Absolute saliency mass inside GT.

        raw_saliency_outside_gt
            Absolute saliency mass outside GT.

        raw_saliency_inside_prediction
            Absolute saliency mass inside prediction.

        raw_saliency_overlap_gt_prediction
            Absolute saliency mass inside GT ∩ prediction.
    """
    # ------------------------------------------------------------
    # 1. Convert inputs to numpy arrays
    # ------------------------------------------------------------
    saliency = np.asarray(
        saliency_map,
        dtype=np.float32
    )
    gt = np.asarray(
        ground_truth_mask
    )
    pred = np.asarray(
        prediction_mask
    )
    # ------------------------------------------------------------
    # 2. Validate dimensions
    # ------------------------------------------------------------
    if saliency.ndim != 2:
        raise ValueError(
            "saliency_map must have shape (H, W)"
        )
    if gt.shape != saliency.shape:
        raise ValueError(
            "ground_truth_mask and saliency_map "
            "must have the same shape"
        )
    if pred.shape != saliency.shape:
        raise ValueError(
            "prediction_mask and saliency_map "
            "must have the same shape"
        )
    # ------------------------------------------------------------
    # 3. Convert masks to binary
    # ------------------------------------------------------------
    gt = gt > 0
    pred = pred > 0
    # ------------------------------------------------------------
    # 4. Make sure saliency is non-negative
    # ------------------------------------------------------------
    saliency = np.nan_to_num(
        saliency,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    saliency = np.maximum(
        saliency,
        0
    )
    # ------------------------------------------------------------
    # 5. Total saliency mass
    # ------------------------------------------------------------
    total_saliency = saliency.sum()
    if total_saliency == 0:
        return {
            "saliency_inside_gt": 0.0,
            "saliency_outside_gt": 0.0,
            "saliency_inside_prediction": 0.0,
            "saliency_overlap_gt_prediction": 0.0,

            "raw_saliency_inside_gt": 0.0,
            "raw_saliency_outside_gt": 0.0,
            "raw_saliency_inside_prediction": 0.0,
            "raw_saliency_overlap_gt_prediction": 0.0,

            "total_saliency": 0.0,
        }
    # ------------------------------------------------------------
    # 6. Define regions
    # ------------------------------------------------------------
    gt_region = gt
    outside_gt_region = ~gt
    prediction_region = pred
    overlap_region = gt & pred
    # ------------------------------------------------------------
    # 7. Calculate raw saliency mass
    # ------------------------------------------------------------
    saliency_inside_gt = saliency[
        gt_region
    ].sum()
    saliency_outside_gt = saliency[
        outside_gt_region
    ].sum()
    saliency_inside_prediction = saliency[
        prediction_region
    ].sum()
    saliency_overlap_gt_prediction = saliency[
        overlap_region
    ].sum()
    # ------------------------------------------------------------
    # 8. Normalize by total saliency
    # ------------------------------------------------------------

    fraction_inside_gt = (
        saliency_inside_gt
        / total_saliency
    )
    fraction_outside_gt = (
        saliency_outside_gt
        / total_saliency
    )
    fraction_inside_prediction = (
        saliency_inside_prediction
        / total_saliency
    )
    fraction_overlap = (
        saliency_overlap_gt_prediction
        / total_saliency
    )
    # ------------------------------------------------------------
    # 9. Return
    # ------------------------------------------------------------
    return {
        # Fractions
        "saliency_inside_gt":
            float(fraction_inside_gt),
        "saliency_outside_gt":
            float(fraction_outside_gt),
        "saliency_inside_prediction":
            float(fraction_inside_prediction),
        "saliency_overlap_gt_prediction":
            float(fraction_overlap),
        # Raw values
        "raw_saliency_inside_gt":
            float(saliency_inside_gt),
        "raw_saliency_outside_gt":
            float(saliency_outside_gt),
        "raw_saliency_inside_prediction":
            float(saliency_inside_prediction),
        "raw_saliency_overlap_gt_prediction":
            float(saliency_overlap_gt_prediction),
        "total_saliency":
            float(total_saliency),
    }

