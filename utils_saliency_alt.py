import torch


def box_iou(boxes1, boxes2):
    """
    Calculate pairwise IoU between two sets of boxes.

    Boxes must be in [x1, y1, x2, y2] format.

    Parameters
    ----------
    boxes1 : Tensor [N, 4]
    boxes2 : Tensor [M, 4]

    Returns
    -------
    iou : Tensor [N, M]
    """

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (len(boxes1), len(boxes2)),
            dtype=torch.float32
        )

    # Intersection
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

    # Areas
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

    # Union
    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / union.clamp(min=1e-8)


def classify_tree_predictions(
    target,
    output,
    tau=0.5
):
    """
    Classify predictions into:

        1. Accurate
        2. Over-segmentation
        3. Under-segmentation
        4. False positive
        5. False negative

    Parameters
    ----------
    target : dict
        Ground truth for ONE image:

        {
            "boxes": Tensor [N_gt, 4],
            "labels": Tensor [N_gt],
            "masks": Tensor [N_gt, 1, H, W],
            ...
        }

    output : dict
        Model predictions for ONE image:

        {
            "boxes": Tensor [N_pred, 4],
            "labels": Tensor [N_pred],
            "scores": Tensor [N_pred],
            "masks": Tensor [N_pred, 1, H, W]
        }

    tau : float
        IoU threshold.

    Returns
    -------
    results : dict
    """

    # =========================================================
    # 1. Extract boxes
    # =========================================================

    gt_boxes = target["boxes"].detach().cpu().float()
    pred_boxes = output["boxes"].detach().cpu().float()

    # Handle no GTs
    if gt_boxes.numel() == 0:
        gt_boxes = torch.empty(
            (0, 4),
            dtype=torch.float32
        )

    # Handle no predictions
    if pred_boxes.numel() == 0:
        pred_boxes = torch.empty(
            (0, 4),
            dtype=torch.float32
        )

    # =========================================================
    # 2. Number of instances
    # =========================================================

    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    # =========================================================
    # 3. Calculate IoU matrix
    #
    # Rows    = GT
    # Columns = Predictions
    #
    #             P1    P2    P3
    # GT1       0.82  0.03  0.00
    # GT2       0.71  0.05  0.00
    # GT3       0.02  0.79  0.01
    #
    # =========================================================

    iou_matrix = box_iou(
        gt_boxes,
        pred_boxes
    )

    # =========================================================
    # 4. Determine relationships
    # =========================================================

    overlap_matrix = iou_matrix >= tau

    # For each GT:
    # which predictions overlap it?
    gt_matches = [
        torch.where(
            overlap_matrix[i]
        )[0].tolist()
        for i in range(n_gt)
    ]

    # For each prediction:
    # which GTs overlap it?
    pred_matches = [
        torch.where(
            overlap_matrix[:, j]
        )[0].tolist()
        for j in range(n_pred)
    ]

    # =========================================================
    # 5. ACCURATE
    #
    # One GT <-> one prediction
    # =========================================================

    accurate = []

    for gt_idx, pred_idxs in enumerate(gt_matches):

        if len(pred_idxs) == 1:

            pred_idx = pred_idxs[0]

            if len(pred_matches[pred_idx]) == 1:

                accurate.append({
                    "gt_index": gt_idx,
                    "pred_index": pred_idx,
                    "iou": float(
                        iou_matrix[
                            gt_idx,
                            pred_idx
                        ]
                    )
                })

    # =========================================================
    # 6. OVER-SEGMENTATION
    #
    # One GT -> multiple predictions
    #
    # GT1 -> P1
    #     -> P2
    #
    # =========================================================

    over_segmentations = []

    for gt_idx, pred_idxs in enumerate(gt_matches):

        if len(pred_idxs) > 1:

            over_segmentations.append({
                "gt_index": gt_idx,
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
    # 7. UNDER-SEGMENTATION
    #
    # Multiple GTs -> one prediction
    #
    # GT1 \
    # GT2  -> P1
    # GT3 /
    #
    # =========================================================

    under_segmentations = []

    for pred_idx, gt_idxs in enumerate(pred_matches):

        if len(gt_idxs) > 1:

            under_segmentations.append({
                "pred_index": pred_idx,
                "gt_indices": gt_idxs,
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
    # 8. FALSE NEGATIVES
    #
    # GT with no prediction above tau
    # =========================================================

    false_negatives = []

    for gt_idx, pred_idxs in enumerate(gt_matches):

        if len(pred_idxs) == 0:

            false_negatives.append({
                "gt_index": gt_idx
            })

    # =========================================================
    # 9. FALSE POSITIVES
    #
    # Prediction with no GT above tau
    # =========================================================

    false_positives = []

    scores = output.get("scores", None)

    if scores is not None:
        scores = scores.detach().cpu()

    for pred_idx, gt_idxs in enumerate(pred_matches):

        if len(gt_idxs) == 0:

            fp = {
                "pred_index": pred_idx
            }

            if scores is not None:
                fp["score"] = float(
                    scores[pred_idx]
                )

            false_positives.append(fp)

    # =========================================================
    # 10. Return
    # =========================================================

    return {
        "accurate": accurate,
        "over_segmentations": over_segmentations,
        "under_segmentations": under_segmentations,
        "false_positives": false_positives,
        "false_negatives": false_negatives,

        # Useful for debugging / analysis
        "iou_matrix": iou_matrix,
        "gt_matches": gt_matches,
        "pred_matches": pred_matches,
    }