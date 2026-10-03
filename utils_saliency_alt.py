import torch
from typing import Dict, Any


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """Pairwise IoU for boxes in [x1, y1, x2, y2] format."""

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (len(boxes1), len(boxes2)),
            dtype=torch.float32
        )

    lt = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2]
    )

    rb = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:]
    )

    wh = (rb - lt).clamp(min=0)

    intersection = wh[..., 0] * wh[..., 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        *
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        *
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / union.clamp(min=1e-8)


def classify_tree_predictions(
    gt_output,
    model_output,
    tau=0.5
):
    """
    Classify tree-crown predictions into:

        accurate
        over-segmentation
        under-segmentation
        false positives
        false negatives

    Supports:

    1. GT as a dictionary containing "boxes"

    OR

    2. GT as a COCO-style list containing dictionaries
       with "bbox".

    Model output should contain:

        "boxes"
        "labels"
        "scores"
        "masks"

    Prediction boxes are assumed to be [x1,y1,x2,y2].

    COCO GT boxes are assumed to be [x,y,width,height].
    """

    # =========================================================
    # 1. Extract GT boxes
    # =========================================================

    gt_ids = []

    # ---------------------------------------------------------
    # CASE A:
    # GT is already a dictionary with "boxes"
    # ---------------------------------------------------------

    if isinstance(gt_output, dict) and "boxes" in gt_output:

        gt_boxes = torch.as_tensor(
            gt_output["boxes"],
            dtype=torch.float32
        ).cpu()

        # Try to get IDs if available
        if "ids" in gt_output:
            gt_ids = gt_output["ids"]

        else:
            gt_ids = list(range(len(gt_boxes)))

    # ---------------------------------------------------------
    # CASE B:
    # GT is a list
    # ---------------------------------------------------------

    elif isinstance(gt_output, (list, tuple)):

        if len(gt_output) == 0:

            gt_boxes = torch.empty(
                (0, 4),
                dtype=torch.float32
            )

            gt_ids = []

        else:

            # -------------------------------------------------
            # Check whether list elements contain "bbox"
            # -------------------------------------------------

            if isinstance(gt_output[0], dict) and "bbox" in gt_output[0]:

                gt_boxes = []

                for i, ann in enumerate(gt_output):

                    x, y, w, h = ann["bbox"]

                    gt_boxes.append([
                        x,
                        y,
                        x + w,
                        y + h
                    ])

                    gt_ids.append(
                        ann.get("id", i)
                    )

                gt_boxes = torch.tensor(
                    gt_boxes,
                    dtype=torch.float32
                )

            else:

                raise ValueError(
                    "GT is a list, but its elements do not "
                    "contain a 'bbox' key. "
                    f"First element is:\n{gt_output[0]}"
                )

    else:

        raise TypeError(
            "gt_output must either be a dictionary "
            "containing 'boxes' or a list of COCO "
            "annotation dictionaries."
        )

    # =========================================================
    # 2. Extract prediction boxes
    # =========================================================

    pred_boxes = torch.as_tensor(
        model_output["boxes"],
        dtype=torch.float32
    ).cpu()

    pred_scores = None

    if "scores" in model_output:

        pred_scores = torch.as_tensor(
            model_output["scores"]
        ).cpu()

    # Handle empty predictions
    if pred_boxes.numel() == 0:

        pred_boxes = torch.empty(
            (0, 4),
            dtype=torch.float32
        )

    # =========================================================
    # 3. Sanity checks
    # =========================================================

    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:

        raise ValueError(
            f"GT boxes must have shape [N,4]. "
            f"Got {gt_boxes.shape}"
        )

    if pred_boxes.ndim != 2 or pred_boxes.shape[1] != 4:

        raise ValueError(
            f"Prediction boxes must have shape [N,4]. "
            f"Got {pred_boxes.shape}"
        )

    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    # =========================================================
    # 4. Calculate IoU matrix
    # =========================================================

    iou_matrix = box_iou(
        gt_boxes,
        pred_boxes
    )

    # =========================================================
    # 5. Establish relationships
    # =========================================================

    overlap_matrix = iou_matrix >= tau

    # For every GT -> predictions
    gt_matches = [
        torch.where(
            overlap_matrix[i]
        )[0].tolist()
        for i in range(n_gt)
    ]

    # For every prediction -> GTs
    pred_matches = [
        torch.where(
            overlap_matrix[:, j]
        )[0].tolist()
        for j in range(n_pred)
    ]

    # =========================================================
    # 6. Accurate
    # =========================================================

    accurate = []

    for gt_idx, pred_idxs in enumerate(gt_matches):

        if len(pred_idxs) == 1:

            pred_idx = pred_idxs[0]

            if len(pred_matches[pred_idx]) == 1:

                accurate.append({
                    "gt_index": gt_idx,
                    "gt_id": gt_ids[gt_idx],
                    "pred_index": pred_idx,
                    "iou": float(
                        iou_matrix[
                            gt_idx,
                            pred_idx
                        ]
                    )
                })

    # =========================================================
    # 7. Over-segmentation
    # =========================================================

    over_segmentations = []

    for gt_idx, pred_idxs in enumerate(gt_matches):

        if len(pred_idxs) > 1:

            over_segmentations.append({
                "gt_index": gt_idx,
                "gt_id": gt_ids[gt_idx],
                "pred_indices": pred_idxs,
                "num_predictions": len(pred_idxs),
                "ious": [
                    float(
                        iou_matrix[
                            gt_idx,
                            pred_idx
                        ]
                    )
                    for pred_idx in pred_idxs
                ]
            })

    # =========================================================
    # 8. Under-segmentation
    # =========================================================

    under_segmentations = []

    for pred_idx, gt_idxs in enumerate(pred_matches):

        if len(gt_idxs) > 1:

            under_segmentations.append({
                "pred_index": pred_idx,
                "gt_indices": gt_idxs,
                "gt_ids": [
                    gt_ids[g]
                    for g in gt_idxs
                ],
                "num_gt": len(gt_idxs),
                "ious": [
                    float(
                        iou_matrix[
                            gt_idx,
                            pred_idx
                        ]
                    )
                    for gt_idx in gt_idxs
                ]
            })

    # =========================================================
    # 9. False negatives
    # =========================================================

    false_negatives = []

    for gt_idx, pred_idxs in enumerate(gt_matches):

        if len(pred_idxs) == 0:

            false_negatives.append({
                "gt_index": gt_idx,
                "gt_id": gt_ids[gt_idx]
            })

    # =========================================================
    # 10. False positives
    # =========================================================

    false_positives = []

    for pred_idx, gt_idxs in enumerate(pred_matches):

        if len(gt_idxs) == 0:

            fp = {
                "pred_index": pred_idx
            }

            if pred_scores is not None:

                fp["score"] = float(
                    pred_scores[pred_idx]
                )

            false_positives.append(fp)

    # =========================================================
    # 11. Return
    # =========================================================

    return {

        "accurate": accurate,

        "over_segmentations":
            over_segmentations,

        "under_segmentations":
            under_segmentations,

        "false_positives":
            false_positives,

        "false_negatives":
            false_negatives,

        "iou_matrix":
            iou_matrix,

        "gt_matches":
            gt_matches,

        "pred_matches":
            pred_matches,
    }