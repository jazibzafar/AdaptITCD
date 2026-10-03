from typing import Dict, List, Any
import torch
import numpy as np
import cv2


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Boxes must be in [x1, y1, x2, y2] format.

    Returns:
        Tensor of shape (N, M)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (len(boxes1), len(boxes2)),
            dtype=torch.float32,
            device=boxes1.device if boxes1.numel() else boxes2.device
        )

    # Intersection
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]

    # Areas
    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    union = area1[:, None] + area2[None, :] - intersection

    return intersection / union.clamp(min=1e-8)


def classify_tree_predictions(
    gt_annos: List[Dict[str, Any]],
    model_output: Dict[str, torch.Tensor],
    tau: float = 0.5,
) -> Dict[str, Any]:
    """
    Classify predictions into accurate, over-segmented,
    under-segmented, false positives and false negatives
    using bounding-box IoU.

    Parameters
    ----------
    gt_annos:
        List of COCO-style annotation dictionaries.
        Each dict should contain at least:
            - 'bbox': [x, y, width, height]
            - 'id'

    model_output:
        Model output dictionary containing:
            - 'boxes': Tensor [B, 4], in [x1, y1, x2, y2] format
            - 'labels'
            - 'scores'
            - 'masks'

    tau:
        IoU threshold used to establish a GT/prediction
        correspondence.

    Returns
    -------
    results:
        Dictionary containing:
            - accurate
            - under_segmentations
            - over_segmentations
            - false_positives
            - false_negatives
            - iou_matrix
            - matched_pairs
    """

    # ---------------------------------------------------------
    # 1. Convert GT COCO boxes
    # COCO format = [x, y, width, height]
    # Convert to [x1, y1, x2, y2]
    # ---------------------------------------------------------

    gt_boxes = []

    for ann in gt_annos:
        x, y, w, h = ann["bbox"]

        gt_boxes.append([
            x,
            y,
            x + w,
            y + h
        ])

    if len(gt_boxes) > 0:
        gt_boxes = torch.tensor(
            gt_boxes,
            dtype=torch.float32
        )
    else:
        gt_boxes = torch.empty(
            (0, 4),
            dtype=torch.float32
        )

    pred_boxes = model_output["boxes"].detach().cpu().float()

    # ---------------------------------------------------------
    # 2. Calculate IoU matrix
    #
    # rows = GT
    # columns = predictions
    # ---------------------------------------------------------

    iou_matrix = box_iou(gt_boxes, pred_boxes)

    # ---------------------------------------------------------
    # 3. Determine all GT-prediction relationships
    #    satisfying IoU >= tau
    # ---------------------------------------------------------

    overlap_matrix = iou_matrix >= tau

    gt_matches = [
        torch.where(overlap_matrix[i])[0].tolist()
        for i in range(len(gt_boxes))
    ]

    pred_matches = [
        torch.where(overlap_matrix[:, j])[0].tolist()
        for j in range(len(pred_boxes))
    ]

    # ---------------------------------------------------------
    # 4. Identify accurate / over / under relationships
    # ---------------------------------------------------------

    accurate = []
    over_segmentations = []
    under_segmentations = []

    # ---- GT perspective ----
    #
    # 1 GT -> 1 prediction = candidate accurate
    # 1 GT -> >1 predictions = over-segmentation

    for gt_idx, pred_idxs in enumerate(gt_matches):

        if len(pred_idxs) > 1:
            over_segmentations.append({
                "gt_index": gt_idx,
                "gt_id": gt_annos[gt_idx]["id"],
                "pred_indices": pred_idxs,
                "num_predictions": len(pred_idxs),
                "ious": [
                    float(iou_matrix[gt_idx, p])
                    for p in pred_idxs
                ]
            })

    # ---- Prediction perspective ----
    #
    # 1 prediction -> >1 GTs = under-segmentation

    for pred_idx, gt_idxs in enumerate(pred_matches):

        if len(gt_idxs) > 1:
            under_segmentations.append({
                "pred_index": pred_idx,
                "gt_indices": gt_idxs,
                "gt_ids": [
                    gt_annos[g]["id"]
                    for g in gt_idxs
                ],
                "num_gt": len(gt_idxs),
                "ious": [
                    float(iou_matrix[g, pred_idx])
                    for g in gt_idxs
                ]
            })

    # ---------------------------------------------------------
    # 5. Accurate = exactly one GT and exactly one prediction
    #
    # Note: this requires a one-to-one relationship.
    # ---------------------------------------------------------

    for gt_idx, pred_idxs in enumerate(gt_matches):

        if len(pred_idxs) == 1:

            pred_idx = pred_idxs[0]

            # Check that prediction also only overlaps this GT
            if len(pred_matches[pred_idx]) == 1:

                accurate.append({
                    "gt_index": gt_idx,
                    "gt_id": gt_annos[gt_idx]["id"],
                    "pred_index": pred_idx,
                    "iou": float(iou_matrix[gt_idx, pred_idx])
                })

    # ---------------------------------------------------------
    # 6. False negatives
    #
    # GTs with NO prediction having IoU >= tau
    # ---------------------------------------------------------

    false_negatives = []

    for gt_idx, pred_idxs in enumerate(gt_matches):

        if len(pred_idxs) == 0:

            false_negatives.append({
                "gt_index": gt_idx,
                "gt_id": gt_annos[gt_idx]["id"]
            })

    # ---------------------------------------------------------
    # 7. False positives
    #
    # Predictions with NO GT having IoU >= tau
    # ---------------------------------------------------------

    false_positives = []

    for pred_idx, gt_idxs in enumerate(pred_matches):

        if len(gt_idxs) == 0:

            false_positives.append({
                "pred_index": pred_idx,
                "score": float(model_output["scores"][pred_idx])
            })

    # ---------------------------------------------------------
    # 8. Return
    # ---------------------------------------------------------

    return {
        "accurate": accurate,
        "over_segmentations": over_segmentations,
        "under_segmentations": under_segmentations,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "iou_matrix": iou_matrix,
        "gt_matches": gt_matches,
        "pred_matches": pred_matches,
    }


def process_saliency_map(
    saliency_map,
    kappa=0.8,
    min_component_area=10,
):
    """
    Keep only the connected saliency component containing the
    global maximum saliency pixel.

    Returns only a single salient object.

    Parameters
    ----------
    saliency_map : np.ndarray
        2D saliency map (H, W)

    kappa : float
        Threshold after normalization.

    min_component_area : int
        Remove components smaller than this area.

    Returns
    -------
    dict
    """
    saliency = np.asarray(saliency_map, dtype=np.float32)
    if saliency.ndim != 2:
        raise ValueError("saliency_map must be 2D")

    # 1. Normalize original saliency to [0,1]
    # s_min = saliency.min()
    # s_max = saliency.max()
    #
    # if s_max > s_min:
    #     saliency_norm = (saliency - s_min) / (s_max - s_min)
    # else:
    #     saliency_norm = np.zeros_like(saliency)

    # 2. Threshold
    saliency_norm = saliency  # just because I commented out the normalization
    binary_mask = (saliency_norm >= kappa).astype(np.uint8)

    # 3. Connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Remove small components
    cleaned_mask = np.zeros_like(binary_mask)

    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned_mask[labels == label] = 1
    # Recompute components after cleaning
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

    # 4. Find global maximum saliency pixel
    peak_idx = np.argmax(saliency_norm)

    peak_y, peak_x = np.unravel_index(peak_idx, saliency_norm.shape)
    peak_value = float(saliency_norm[peak_y, peak_x])

    # 5. Select only component containing peak
    peak_label = labels[peak_y, peak_x]
    selected_mask = np.zeros_like(binary_mask)
    if peak_label != 0:
        selected_mask[labels == peak_label] = 1

    # 6. Create single-component thresholded saliency
    selected_thresholded = (saliency_norm * selected_mask)

    # 7. Polygon
    contours, _ = cv2.findContours(
        selected_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygon = None
    if len(contours) > 0:
        polygon = (contours[0][:, 0, :].astype(np.int32))

    # 8. Normalize selected component
    normalized_saliency = np.zeros_like(saliency_norm)
    values = selected_thresholded[selected_mask > 0]

    if values.size > 0:
        vmin = values.min()
        vmax = values.max()
        if vmax > vmin:
            normalized_saliency[selected_mask > 0] = ((values - vmin) / (vmax - vmin))
        else:
            normalized_saliency[selected_mask > 0] = 1.0

    # 9. Return ONLY selected component
    return {
        # original normalized map
        "saliency_map": saliency_norm,
        # ONLY one component
        "thresholded_map": selected_thresholded,
        # ONLY one component
        "saliency_mask": selected_mask,
        # polygon mask
        "polygon_mask": selected_mask,
        # single polygon
        "polygon": polygon,
        # normalized selected saliency
        "normalized_saliency": normalized_saliency,
        "peak": (
            int(peak_x),
            int(peak_y)
        ),
        "peak_value": peak_value,
        # debugging
        "num_components_before_selection": n_labels - 1,
    }


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
    saliency = np.asarray(saliency_map, dtype=np.float32)
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


def masks_to_coco_polygons(masks, threshold=0.5):
    """
    Convert masks of shape (N, 1, H, W) into COCO polygon format.

    Returns:
        list[list[list[float]]]:
            One COCO segmentation per mask.
    """
    coco_segmentations = []

    for mask in masks:
        # (1, H, W) -> (H, W)
        mask = mask.squeeze(0)

        # If mask contains probabilities, threshold it
        binary_mask = (mask >= threshold).astype(np.uint8)

        # Find external contours
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []

        for contour in contours:
            # COCO polygons need at least 3 points
            if len(contour) < 3:
                continue

            # (N, 1, 2) -> (N, 2)
            contour = contour.squeeze(1)

            # Flatten [(x,y), (x,y), ...]
            polygon = contour.flatten().tolist()

            # Need at least 3 vertices = 6 coordinates
            if len(polygon) >= 6:
                polygons.append(polygon)

        coco_segmentations.append(polygons)

    return coco_segmentations
