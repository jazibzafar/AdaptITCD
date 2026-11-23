##
from utils import yaml_to_dict_parser
from tifffile import imread
import matplotlib.pyplot as plt
import numpy as np
from test_reading_ckpt import load_pretrained_maskrcnn_model, plot_annotations
from tifffile import imread
from inference import match_rgb_histogram_with_exclusions, collect_preds
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize
import torch
import gc


model_path = "/home/jazib/projects/savedmodels/s1_convnext_full_full"
model = load_pretrained_maskrcnn_model(model_path)

tif = imread("/home/jazib/projects/data/MarieFlambeck/newtiles/tile_00004.tif")
hist_ali = yaml_to_dict_parser("./preds/oam_biome_aligned_4_5_aligned_histcdf.yaml")
hist_gen = yaml_to_dict_parser("./preds/oam_general_histcdf.yaml")


ref_hist_ali = np.vstack([hist_ali["hist"]["red"],
                          hist_ali["hist"]["green"],
                          hist_ali["hist"]["blue"]])

ref_hist_gen = np.vstack([hist_gen["hist"]["red"],
                          hist_gen["hist"]["green"],
                          hist_gen["hist"]["blue"]])

img_ali = match_rgb_histogram_with_exclusions(tif, ref_hist_ali, exclude_values=[0, 255])
img_gen = match_rgb_histogram_with_exclusions(tif, ref_hist_gen, exclude_values=[0, 255])

OAM_NORM_MEAN = [0.452, 0.466, 0.364]
OAM_NORM_STDV = [0.235, 0.227, 0.220]

ten_img = F.to_tensor(tif)
ten_img_ali = F.to_tensor(img_ali)
ten_img_gen = F.to_tensor(img_gen)


norm = Normalize(
    mean=OAM_NORM_MEAN,
    std=OAM_NORM_STDV
)

ten_img_norm = norm(ten_img)
ten_img_ali_norm = norm(ten_img_ali)
ten_img_gen_norm = norm(ten_img_gen)

##
# model.to("cuda")
# model.eval()
#
# with torch.no_grad():
#     out_img = model([ten_img.to("cuda")])
#     out_img_ali = model([ten_img_ali.to("cuda")])
#     out_img_gen = model([ten_img_gen.to("cuda")])
#     out_img_norm = model([ten_img_norm.to("cuda")])
#     out_img_ali_norm = model([ten_img_ali_norm.to("cuda")])
#     out_img_gen_norm = model([ten_img_gen_norm.to("cuda")])
#
#
# preds_img_ali_norm = collect_preds(out_img_ali_norm, score_threshold=0.01, filter_tree_groups=True)
# preds_img_gen_norm = collect_preds(out_img_gen_norm, score_threshold=0.01, filter_tree_groups=True)
# preds_img_ali = collect_preds(out_img_ali, score_threshold=0.01, filter_tree_groups=True)
# preds_img = collect_preds(out_img, score_threshold=0.01, filter_tree_groups=True)
#
#
# plot_annotations(
#     image=img_ali,
#     annotations=preds_img_ali_norm[0]["annotations"]
# )

##
# import torch
from torchvision.ops import batched_nms, box_iou


# ---------------------------------------------------------
# 1. TTA transformations
# ---------------------------------------------------------

def apply_tta(image, transform):

    if transform == "identity":
        return image

    elif transform == "hflip":
        return torch.flip(image, dims=[-1])

    elif transform == "vflip":
        return torch.flip(image, dims=[-2])

    elif transform == "hvflip":
        return torch.flip(image, dims=[-2, -1])

    raise ValueError(f"Unknown transform: {transform}")


def invert_tta_boxes(boxes, transform, height, width):

    boxes = boxes.clone()

    if transform in ("hflip", "hvflip"):
        x1 = width - boxes[:, 2]
        x2 = width - boxes[:, 0]

        boxes[:, 0] = x1
        boxes[:, 2] = x2

    if transform in ("vflip", "hvflip"):
        y1 = height - boxes[:, 3]
        y2 = height - boxes[:, 1]

        boxes[:, 1] = y1
        boxes[:, 3] = y2

    return boxes


def invert_tta_masks(masks, transform):

    if transform == "hflip":
        return torch.flip(masks, dims=[-1])

    elif transform == "vflip":
        return torch.flip(masks, dims=[-2])

    elif transform == "hvflip":
        return torch.flip(masks, dims=[-2, -1])

    return masks


# ---------------------------------------------------------
# 2. Prediction merging
# ---------------------------------------------------------

def merge_predictions(
    predictions,
    nms_threshold=0.5,
):
    """
    Merge multiple Mask R-CNN predictions using
    class-aware box NMS.

    predictions: List of dictionaries containing
                 boxes, labels, scores and masks.
    """

    if not predictions:
        return None

    boxes = torch.cat([p["boxes"] for p in predictions])
    labels = torch.cat([p["labels"] for p in predictions])
    scores = torch.cat([p["scores"] for p in predictions])
    masks = torch.cat([p["masks"] for p in predictions])

    if len(boxes) == 0:
        return None

    keep = batched_nms(
        boxes,
        scores,
        labels,
        nms_threshold
    )

    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
        "scores": scores[keep],
        "masks": masks[keep],
    }


# ---------------------------------------------------------
# 3. Remove detections already found in previous iterations
# ---------------------------------------------------------

def filter_previous_detections(
    prediction,
    previous_boxes,
    previous_labels,
    iou_threshold=0.5,
):
    """
    Remove predictions that overlap with detections
    from previous iterations.

    Only detections of the same class are compared.
    """

    if len(previous_boxes) == 0:
        return prediction

    boxes = prediction["boxes"]
    labels = prediction["labels"]

    iou = box_iou(boxes, previous_boxes)

    same_class = labels[:, None] == previous_labels[None, :]

    duplicate = ((iou >= iou_threshold) & same_class).any(dim=1)

    keep = ~duplicate

    return {
        k: v[keep] for k, v in prediction.items()
    }


# ---------------------------------------------------------
# 4. Iterative Mask R-CNN with TTA
# ---------------------------------------------------------

@torch.inference_mode()
def iterative_mask_rcnn(
    model,
    image,
    max_iterations=3,
    transforms=("identity", "hflip", "vflip", "hvflip"),
    score_threshold=0.5,
    mask_threshold=0.5,
    tree_group_label=1,
    tta_nms_threshold=0.5,
    global_nms_threshold=0.5,
    duplicate_iou_threshold=0.5,
    mask_value=0.0,
):
    """
    Iterative Mask R-CNN inference with TTA.

    Args:
        model:
            Torchvision Mask R-CNN.

        image:
            Tensor [C, H, W], already normalized.

        max_iterations:
            Maximum number of iterative inference passes.

        transforms:
            TTA transformations.

        score_threshold:
            Minimum detection confidence.

        mask_threshold:
            Threshold for binarizing predicted masks.

        tree_group_label:
            Class ID to exclude (default: 1).

        tta_nms_threshold:
            NMS threshold for merging TTA predictions.

        global_nms_threshold:
            Final NMS threshold across all iterations.

        duplicate_iou_threshold:
            IoU threshold for removing detections
            already found in previous iterations.

        mask_value:
            Value used to replace detected crown pixels.
            0.0 corresponds to the dataset mean in normalized
            image space.

    Returns:
        Dictionary containing:
            boxes
            labels
            scores
            masks
    """
    model.to("cuda")
    model.eval()

    _, height, width = image.shape

    current_image = image.clone()

    all_predictions = []

    previous_boxes = []
    previous_labels = []

    # -----------------------------------------------------
    # Iterative inference
    # -----------------------------------------------------

    for iteration in range(max_iterations):

        iteration_predictions = []

        # -------------------------------------------------
        # TTA inference
        # -------------------------------------------------

        for transform in transforms:

            augmented_image = apply_tta(
                current_image,
                transform
            )

            prediction = model([augmented_image])[0]

            # Invert transformations
            boxes = invert_tta_boxes(
                prediction["boxes"],
                transform,
                height,
                width
            )

            masks = invert_tta_masks(
                prediction["masks"],
                transform
            )

            # Filter confidence and tree-group class
            keep = (
                (prediction["scores"] >= score_threshold)
                &
                (prediction["labels"] != tree_group_label)
            )

            if not keep.any():
                continue

            iteration_predictions.append({
                "boxes": boxes[keep],
                "labels": prediction["labels"][keep],
                "scores": prediction["scores"][keep],
                "masks": masks[keep],
            })

        # -------------------------------------------------
        # Merge TTA predictions
        # -------------------------------------------------

        iteration_result = merge_predictions(
            iteration_predictions,
            nms_threshold=tta_nms_threshold
        )

        if iteration_result is None:
            break

        # -------------------------------------------------
        # Remove detections from previous iterations
        # -------------------------------------------------

        if previous_boxes:

            iteration_result = filter_previous_detections(
                iteration_result,
                torch.cat(previous_boxes),
                torch.cat(previous_labels),
                iou_threshold=duplicate_iou_threshold
            )

        if len(iteration_result["boxes"]) == 0:
            break

        # -------------------------------------------------
        # Collect new detections
        # -------------------------------------------------

        all_predictions.append(iteration_result)

        previous_boxes.append(iteration_result["boxes"])
        previous_labels.append(iteration_result["labels"])

        # -------------------------------------------------
        # Mask newly detected crowns
        # -------------------------------------------------

        binary_masks = (
            iteration_result["masks"][:, 0] >= mask_threshold
        )

        mask_region = binary_masks.any(dim=0)

        if not mask_region.any():
            break

        # Support either scalar or per-channel mask values
        if isinstance(mask_value, torch.Tensor):
            fill_value = mask_value.to(
                device=image.device,
                dtype=image.dtype
            ).view(-1, 1)
        else:
            fill_value = mask_value

        current_image[:, mask_region] = fill_value

    # -----------------------------------------------------
    # Global merging across all iterations
    # -----------------------------------------------------

    final_prediction = merge_predictions(
        all_predictions,
        nms_threshold=global_nms_threshold
    )

    if final_prediction is None:
        return {
            "boxes": image.new_empty((0, 4)),
            "labels": torch.empty(
                0, dtype=torch.int64, device=image.device
            ),
            "scores": image.new_empty((0,)),
            "masks": image.new_empty((0, 1, height, width)),
        }

    # Release temporary tensors
    del current_image
    del all_predictions
    del previous_boxes
    del previous_labels

    gc.collect()
    torch.cuda.empty_cache()
    return final_prediction


##
out_iter = iterative_mask_rcnn(model=model,
                               image=ten_img_ali_norm.to("cuda"),
                               max_iterations=1,
                               score_threshold=0.05,
                               tta_nms_threshold=0.3,
                               global_nms_threshold=0.3,
                               duplicate_iou_threshold=0.3,
                               mask_value=0)
out_img_ali_norm = collect_preds([out_iter], score_threshold=0.01, filter_tree_groups=True)
plot_annotations(img_ali, out_img_ali_norm[0]["annotations"])

##
from inference import ensemble_mask_nms


ensemble_output = ensemble_mask_nms(
    outputs=[
        out_img_ali,
        out_img_gen,
        out_img_ali_norm,
    ],
    image_size=(1024, 1024),
    iou_threshold=0.5,
    score_threshold=0.3,
    class_agnostic=False,
)

##
from matplotlib.patches import Rectangle, Polygon


def plot_annotations_save(image, annotations, save_path):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    for anno in annotations:
        # Draw Masks
        polygon = anno['segmentation'][0]
        polygon = np.asarray(polygon).reshape(-1, 2)
        patch = Polygon(
            polygon,
            closed=True,
            facecolor="blue",
            edgecolor="none",
            alpha=0.5,
        )
        ax.add_patch(patch)
        # Draw Boxes
        box = anno['box']
        x1, y1, x2, y2 = box
        rect = Rectangle(
            (x1, y1),
            x2-x1,
            y2-y1,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.axis("off")
    # plt.savefig()


##
##
from torchvision.ops import batched_nms


def apply_tta(image, transform):
    """
    image: Tensor [C, H, W]
    """
    if transform == "identity":
        return image
    elif transform == "hflip":
        return torch.flip(image, dims=[-1])
    elif transform == "vflip":
        return torch.flip(image, dims=[-2])
    elif transform == "hvflip":
        return torch.flip(image, dims=[-2, -1])

    raise ValueError(f"Unknown transform: {transform}")


def invert_boxes(boxes, transform, height, width):
    """
    Transform predicted boxes back to original image coordinates.

    boxes: Tensor [N, 4], xyxy format.
    """
    boxes = boxes.clone()

    if transform in ("hflip", "hvflip"):
        x1 = width - boxes[:, 2]
        x2 = width - boxes[:, 0]
        boxes[:, 0] = x1
        boxes[:, 2] = x2

    if transform in ("vflip", "hvflip"):
        y1 = height - boxes[:, 3]
        y2 = height - boxes[:, 1]
        boxes[:, 1] = y1
        boxes[:, 3] = y2

    return boxes


def invert_masks(masks, transform):
    """
    masks: Tensor [N, 1, H, W]
    """
    if transform == "hflip":
        return torch.flip(masks, dims=[-1])
    elif transform == "vflip":
        return torch.flip(masks, dims=[-2])
    elif transform == "hvflip":
        return torch.flip(masks, dims=[-2, -1])
    return masks


@torch.inference_mode()
def predict_tta(
    model,
    image,
    transforms=("identity", "hflip", "vflip", "hvflip"),
    nms_threshold=0.5,
    score_threshold=0.05,
):
    """
    Test-time augmentation for torchvision Mask R-CNN.

    Args:
        model: Torchvision Mask R-CNN in eval mode.
        image: Tensor [C, H, W].
        transforms: List of transformations.
        nms_threshold: IoU threshold for merging detections.
        score_threshold: Minimum detection confidence.

    Returns:
        Dictionary containing:
            boxes: [N, 4]
            labels: [N]
            scores: [N]
            masks: [N, 1, H, W]
    """

    model.eval()
    model.to("cuda")

    _, height, width = image.shape

    all_boxes = []
    all_labels = []
    all_scores = []
    all_masks = []

    for transform in transforms:

        # Transform image
        augmented_image = apply_tta(image, transform)

        # Model inference
        prediction = model([augmented_image.to("cuda")])[0]

        # Invert bounding boxes
        boxes = invert_boxes(
            prediction["boxes"],
            transform,
            height,
            width
        )

        # Invert segmentation masks
        masks = invert_masks(
            prediction["masks"],
            transform
        )

        # Filter low-confidence predictions
        keep = prediction["scores"] >= score_threshold

        all_boxes.append(boxes[keep])
        all_labels.append(prediction["labels"][keep])
        all_scores.append(prediction["scores"][keep])
        all_masks.append(masks[keep])

    # Concatenate predictions
    boxes = torch.cat(all_boxes, dim=0)
    labels = torch.cat(all_labels, dim=0)
    scores = torch.cat(all_scores, dim=0)
    masks = torch.cat(all_masks, dim=0)

    # Class-aware NMS
    keep = batched_nms(
        boxes,
        scores,
        labels,
        nms_threshold
    )

    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
        "scores": scores[keep],
        "masks": masks[keep],
    }


##
out_tta = predict_tta(model=model, image=ten_img_ali_norm)
p_tta = collect_preds([out_tta], score_threshold=0.01, filter_tree_groups=True)
plot_annotations(img_ali, p_tta[0]["annotations"])
