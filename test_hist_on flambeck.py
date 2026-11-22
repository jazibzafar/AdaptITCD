##
from utils import yaml_to_dict_parser
from tifffile import imread
import matplotlib.pyplot as plt
import numpy as np


def match_rgb_histogram_with_exclusions(target_image, ref_hist, exclude_values=None):
    """
    Matches an RGB target image histogram to a reference histogram while excluding specific pixel values.

    Parameters:
    - target_image: 3D NumPy array (H, W, 3) representing an RGB image.
    - ref_hist: NumPy array of size 768 (concatenated RGB) or shape (3, 256).
    - exclude_values: List or tuple of pixel values to ignore (e.g., [0, 255] or [0, 1]).

    Returns:
    - matched_image: 3D NumPy array with histogram matched, leaving excluded values untouched.
    """
    if exclude_values is None:
        exclude_values = []

    target_image = np.asarray(target_image)

    # Handle float images normalized to [0, 1]
    is_float = np.issubdtype(target_image.dtype, np.floating)
    if is_float and target_image.max() <= 1.0:
        target_image = (target_image * 255).astype(np.uint8)

    original_dtype = target_image.dtype

    # Format reference histogram into shape (3, 256)
    ref_hist = np.asarray(ref_hist, dtype=np.float64)
    if ref_hist.size == 768:
        ref_hist = ref_hist.reshape(3, 256)
    elif ref_hist.shape == (256, 3):
        ref_hist = ref_hist.T

    matched_channels = []

    for i in range(3):
        channel_hist = ref_hist[i].copy()
        num_bins = len(channel_hist)  # 256
        target_channel = target_image[..., i]

        # 1. Zero out excluded bins in the reference histogram
        for val in exclude_values:
            if 0 <= val < num_bins:
                channel_hist[int(val)] = 0

        # 2. Compute target channel histogram and zero out excluded bins
        target_h, _ = np.histogram(target_channel.flatten(), bins=num_bins, range=(0, num_bins))
        for val in exclude_values:
            if 0 <= val < num_bins:
                target_h[int(val)] = 0

        # 3. Compute normalized CDFs using only the valid/foreground bins
        target_cdf = np.cumsum(target_h).astype(np.float64)
        if target_cdf[-1] > 0:
            target_cdf /= target_cdf[-1]

        ref_cdf = np.cumsum(channel_hist)
        if ref_cdf[-1] > 0:
            ref_cdf /= ref_cdf[-1]
        else:
            # Fallback linear CDF if all bins are excluded
            ref_cdf = np.linspace(0, 1, num_bins)

        # 4. Interpolate mapping table
        ref_pixels = np.arange(num_bins, dtype=np.float64)
        mapping = np.interp(target_cdf, ref_cdf, ref_pixels)

        # 5. Map the pixels
        pixel_idxs = target_channel.clip(0, num_bins - 1).astype(int)
        matched_channel = mapping[pixel_idxs]

        # 6. Force excluded values in the target image to remain unchanged (map to themselves)
        for val in exclude_values:
            matched_channel[target_channel == val] = val

        matched_channels.append(np.clip(matched_channel, 0, num_bins - 1))

    matched_image = np.stack(matched_channels, axis=-1)
    return matched_image.astype(original_dtype)


##
from test_reading_ckpt import load_pretrained_maskrcnn_model


model = load_pretrained_maskrcnn_model("/home/jazib/projects/savedmodels/s1_convnext_full_full")
##
from tifffile import imread


tif = imread("/home/jazib/projects/data/MarieFlambeck/newtiles/tile_00003.tif")
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

# plt.imshow(tif)
# plt.show()

# plt.imshow(img_gen)
# plt.show()
##
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize


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
import torch


model.to("cuda")
model.eval()

with torch.no_grad():
    out_img = model([ten_img.to("cuda")])
    out_img_ali = model([ten_img_ali.to("cuda")])
    out_img_gen = model([ten_img_gen.to("cuda")])
    out_img_norm = model([ten_img_norm.to("cuda")])
    out_img_ali_norm = model([ten_img_ali_norm.to("cuda")])
    out_img_gen_norm = model([ten_img_gen_norm.to("cuda")])


##
from test_reading_ckpt import plot_annotations, mask_to_polygons

def collect_preds(outputs, score_threshold, filter_tree_groups=True):
    """
    Filter low confidence predictions (based on bbox confidence) and create a dict for each tile.
    :param tile_names: list; names of all the tiles in a batch
    :param outputs: mask rcnn pred outputs
    :param score_threshold: min confidence score to keep the annotation
    :param filter_tree_groups: trigger it true to filter out tree group preds and keep only single tree preds
    :return: dict with key tile_name and values all annotations
    """
    # assert len(tile_names) == len(outputs), "len(tile_names) /= len(outputs)"
    # infer batch size from the size of tile_names and outputs
    batch_size = 1
    pred_list = []
    for i in range(batch_size):
        # current_tile = tile_names[i]  # is a string
        current_output = outputs[i]  # is a dict
        boxes = current_output["boxes"]
        labels = current_output["labels"]
        scores = current_output["scores"]
        masks = current_output["masks"]
        annotations = []
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            # optional label filtering for tree groups:
            if filter_tree_groups:
                if label == 1:
                    continue
            # score filtering
            score = score.detach().cpu().item()
            if score < score_threshold:
                continue
            # getting box, label, segmentation, area in the right shape
            box = box.detach().cpu().numpy()
            label = label.detach().cpu().item()
            # mask shape = [1, H, W]
            mask = mask.detach().cpu().numpy()
            mask = mask[0]
            mask_binary = (mask >= 0.5).astype(np.uint8)
            segmentation = mask_to_polygons(mask_binary)
            if len(segmentation) == 0:
                continue
            area = float(mask_binary.sum())
            # append annotations list
            annotations.append({
                "id": len(annotations) + 1,
                "label": label,
                "box": box.tolist(),
                "segmentation": segmentation,
                "area": area,
                "score": score,
            })
        pred_list.append({
            # "tile": current_tile,
            "num_detections": len(annotations),
            "annotations": annotations
        })
    return pred_list


preds_img_ali_norm = collect_preds(out_img_ali_norm, score_threshold=0.01, filter_tree_groups=True)
preds_img_gen_norm = collect_preds(out_img_gen_norm, score_threshold=0.01, filter_tree_groups=True)
preds_img_ali = collect_preds(out_img_ali, score_threshold=0.01, filter_tree_groups=True)
preds_img = collect_preds(out_img, score_threshold=0.01, filter_tree_groups=True)


plot_annotations(
    image=img_ali,
    annotations=preds_img_ali_norm[0]["annotations"]
)

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
p_tta = predict_tta(model=model, image=ten_img_ali_norm)
##

out_tta = collect_preds([p_tta], score_threshold=0.01, filter_tree_groups=True)

##
plot_annotations(img_ali, out_tta[0]["annotations"])


##
from torchvision.ops import box_iou

@torch.inference_mode()
def iterative_mask_rcnn(
    model,
    image,
    max_iterations=3,
    score_threshold=0.5,
    mask_threshold=0.5,
    mask_value=0.0,
    duplicate_iou=0.5,
):
    """
    Iterative object detection by masking previously
    detected instance segmentation masks.

    Args:
        model: Torchvision Mask R-CNN.
        image: Tensor [C, H, W], already normalized.
        max_iterations: Maximum number of inference passes.
        score_threshold: Minimum detection confidence.
        mask_threshold: Binary mask threshold.
        mask_value: Replacement value in normalized space.
        duplicate_iou: IoU threshold for duplicate boxes.

    Returns:
        List of predictions from all iterations.
    """

    model.eval()

    current_image = image.clone()
    all_predictions = []
    previous_boxes = []

    for iteration in range(max_iterations):

        prediction = model([current_image])[0]

        # Confidence filtering
        keep = prediction["scores"] >= score_threshold

        prediction = {
            k: v[keep] for k, v in prediction.items()
        }

        if len(prediction["boxes"]) == 0:
            break

        boxes = prediction["boxes"]

        # Remove detections overlapping previous iterations
        if previous_boxes:

            old_boxes = torch.cat(previous_boxes)

            iou = box_iou(boxes, old_boxes)

            is_duplicate = iou.max(dim=1).values >= duplicate_iou

            keep = ~is_duplicate

            prediction = {
                k: v[keep] for k, v in prediction.items()
            }

        if len(prediction["boxes"]) == 0:
            break

        # Store new detections
        all_predictions.append(prediction)

        previous_boxes.append(prediction["boxes"])

        # Mask newly detected segmentation masks
        masks = prediction["masks"][:, 0] >= mask_threshold

        if masks.any():
            mask_region = masks.any(dim=0)

            current_image[:, mask_region] = mask_value

    return all_predictions


out_mask = iterative_mask_rcnn(model.to("cuda"), ten_img_ali_norm.to("cuda"), max_iterations=5)
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