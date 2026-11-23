import numpy as np
from test_reading_ckpt import mask_to_polygons
from pycocotools import mask as mask_utils


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


def coco_polygon_to_mask(segmentation, height, width):
    """
    Convert COCO polygon segmentation to binary mask.

    Args:
        segmentation: List of COCO polygon coordinate lists.
        height, width: Original image dimensions.

    Returns:
        Binary numpy array [H, W].
    """

    if not segmentation:
        return np.zeros((height, width), dtype=bool)

    rles = mask_utils.frPyObjects(
        segmentation,
        height,
        width
    )

    rle = mask_utils.merge(rles)

    mask = mask_utils.decode(rle)

    return mask.astype(bool)


def mask_iou(mask1, mask2):
    """Calculate binary mask IoU."""

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def ensemble_mask_nms(
    outputs,
    image_size,
    iou_threshold=0.5,
    score_threshold=0.0,
    class_agnostic=False,
):
    """
    Merge predictions from multiple models using mask IoU NMS.

    Args:
        outputs: List of model outputs.
        image_size: (height, width) of original image.
        iou_threshold: Mask IoU suppression threshold.
        score_threshold: Minimum confidence threshold.
        class_agnostic: Whether to ignore class labels.

    Returns:
        {
            'num_detections': int,
            'annotations': [...]
        }
    """

    height, width = image_size

    # -------------------------------------------------
    # Collect annotations
    # -------------------------------------------------

    annotations = []

    for output in outputs:
        for ann in output["annotations"]:

            if ann["score"] < score_threshold:
                continue

            annotations.append(ann)

    if not annotations:
        return {
            "num_detections": 0,
            "annotations": []
        }

    # -------------------------------------------------
    # Sort by confidence (descending)
    # -------------------------------------------------

    order = sorted(
        range(len(annotations)),
        key=lambda i: annotations[i]["score"],
        reverse=True
    )

    # -------------------------------------------------
    # Decode segmentation masks
    # -------------------------------------------------

    masks = [
        coco_polygon_to_mask(
            ann["segmentation"],
            height,
            width
        )
        for ann in annotations
    ]

    # -------------------------------------------------
    # Greedy mask NMS
    # -------------------------------------------------

    keep = []
    suppressed = np.zeros(len(annotations), dtype=bool)

    for idx in order:

        if suppressed[idx]:
            continue

        keep.append(idx)

        current_mask = masks[idx]
        current_label = annotations[idx]["label"]

        for other_idx in order:

            if other_idx == idx or suppressed[other_idx]:
                continue

            # Class-aware suppression
            if not class_agnostic:

                if annotations[other_idx]["label"] != current_label:
                    continue

            iou = mask_iou(
                current_mask,
                masks[other_idx]
            )

            if iou >= iou_threshold:
                suppressed[other_idx] = True

    # -------------------------------------------------
    # Construct final annotations
    # -------------------------------------------------

    final_annotations = []

    for new_id, idx in enumerate(keep):

        ann = annotations[idx].copy()

        ann["id"] = new_id

        final_annotations.append(ann)

    return {
        "num_detections": len(final_annotations),
        "annotations": final_annotations
    }