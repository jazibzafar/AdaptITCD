import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from pycocotools import mask as mask_utils


def polygon_to_mask(segmentation, height, width):
    """
    Convert COCO polygon segmentation to a binary mask.

    segmentation:
        COCO polygon format, e.g.
        [[x1, y1, x2, y2, ...], ...]

    Returns:
        Boolean numpy array of shape (height, width)
    """

    if segmentation is None:
        return None

    rles = mask_utils.frPyObjects(
        segmentation,
        height,
        width
    )

    rle = mask_utils.merge(rles)

    mask = mask_utils.decode(rle)

    if mask.ndim == 3:
        mask = np.any(mask, axis=2)

    return mask.astype(bool)


def polygon_to_rle(segmentation, height, width):
    """Convert COCO polygon to compressed RLE."""

    if segmentation is None:
        return None

    rles = mask_utils.frPyObjects(
        segmentation,
        height,
        width
    )

    return mask_utils.merge(rles)


def build_gt_rle_cache(
    gt_annotations,
    image_sizes
):

    rle_cache = {}

    for idx, ann in enumerate(gt_annotations):

        image_id = ann["image_id"]

        height, width = image_sizes[image_id]

        rle_cache[idx] = polygon_to_rle(
            ann["segmentation"],
            height,
            width
        )

    return rle_cache


def build_rle_cache(annotations, image_sizes):
    """
    Convert prediction polygon segmentations to COCO RLE once.

    Parameters
    ----------
    annotations : list of dict
        Prediction annotations. Each annotation must contain:
        - anno_id
        - image_id
        - segmentation

    image_sizes : dict
        Maps image_id -> (height, width)

    Returns
    -------
    dict
        Maps anno_id -> RLE
    """

    rle_cache = {}

    for ann in annotations:

        image_id = ann["image_id"]

        height, width = image_sizes[image_id]

        segmentation = ann["segmentation"]

        if segmentation is None:
            rle_cache[ann["anno_id"]] = None
            continue

        rles = mask_utils.frPyObjects(
            segmentation,
            height,
            width
        )

        rle = mask_utils.merge(rles)

        rle_cache[ann["anno_id"]] = rle

    return rle_cache


def rle_iou(rle_a, rle_b):

    if rle_a is None or rle_b is None:
        return np.nan

    return float(
        mask_utils.iou(
            [rle_a],
            [rle_b],
            [0]
        )[0, 0]
    )


def calculate_s2_s3_iou_fast(
    instance,
    rle_cache
):

    s2 = instance["s2"]
    s3 = instance["s3"]

    if s2 is None or s3 is None:
        return np.nan

    rle_s2 = rle_cache[s2["anno_id"]]
    rle_s3 = rle_cache[s3["anno_id"]]

    return rle_iou(
        rle_s2,
        rle_s3
    )


def mask_iou(mask_a, mask_b):
    """
    Calculate IoU between two binary masks.
    """

    if mask_a is None or mask_b is None:
        return np.nan

    intersection = np.logical_and(
        mask_a,
        mask_b
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    if union == 0:
        return 0.0

    return intersection / union


def calculate_s2_s3_iou(
    instance,
    height,
    width
):
    """
    Calculate IoU between s2 and s3 masks.
    """

    if (
        instance["s2"] is None
        or instance["s3"] is None
    ):
        return np.nan

    mask_s2 = polygon_to_mask(
        instance["s2"]["segmentation"],
        height,
        width
    )

    mask_s3 = polygon_to_mask(
        instance["s3"]["segmentation"],
        height,
        width
    )

    return mask_iou(mask_s2, mask_s3)


def calculate_mask_disagreement(
    instance,
    height,
    width
):
    """
    Calculate pairwise mask disagreement.

    Returns:
        mean_pairwise_iou
        pairwise_iou_std
        mask_disagreement
        iou_s2_s3
    """

    num_models = instance["num_models"]

    # --------------------------------------------------
    # Three models
    # --------------------------------------------------

    if num_models == 3:

        iou_s1_s2 = instance["iou_s1_s2"]
        iou_s1_s3 = instance["iou_s1_s3"]

        iou_s2_s3 = calculate_s2_s3_iou(
            instance,
            height,
            width
        )

        ious = np.array([
            iou_s1_s2,
            iou_s1_s3,
            iou_s2_s3
        ])

        mean_iou = np.mean(ious)
        std_iou = np.std(ious)

        disagreement = 1.0 - mean_iou

        return (
            mean_iou,
            std_iou,
            disagreement,
            iou_s2_s3
        )

    # --------------------------------------------------
    # Two models
    # --------------------------------------------------

    elif num_models == 2:

        ious = []

        if instance["iou_s1_s2"] is not None:
            ious.append(instance["iou_s1_s2"])

        if instance["iou_s1_s3"] is not None:
            ious.append(instance["iou_s1_s3"])

        if len(ious) == 0:
            return np.nan, np.nan, np.nan, np.nan

        mean_iou = np.mean(ious)
        std_iou = np.std(ious)

        disagreement = 1.0 - mean_iou

        return (
            mean_iou,
            std_iou,
            disagreement,
            np.nan
        )

    # --------------------------------------------------
    # One model
    # --------------------------------------------------

    else:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

def calculate_disagreement_fast(
    instance,
    rle_cache
):

    ious = []

    if instance["s1"] is not None and instance["s2"] is not None:
        ious.append(instance["iou_s1_s2"])

    if instance["s1"] is not None and instance["s3"] is not None:
        ious.append(instance["iou_s1_s3"])

    iou_s2_s3 = np.nan

    if instance["s2"] is not None and instance["s3"] is not None:

        iou_s2_s3 = calculate_s2_s3_iou_fast(
            instance,
            rle_cache
        )

        ious.append(iou_s2_s3)

    if len(ious) < 1:

        return (
            np.nan,
            np.nan,
            np.nan,
            iou_s2_s3
        )

    ious = np.asarray(ious)

    mean_iou = ious.mean()
    std_iou = ious.std()

    disagreement = 1.0 - mean_iou

    return (
        mean_iou,
        std_iou,
        disagreement,
        iou_s2_s3
    )


def bbox_iou_fast(box_a, box_b):

    ax_min, ax_max, ay_min, ay_max = box_a
    bx_min, bx_max, by_min, by_max = box_b

    ix_min = max(ax_min, bx_min)
    ix_max = min(ax_max, bx_max)

    iy_min = max(ay_min, by_min)
    iy_max = min(ay_max, by_max)

    iw = max(0, ix_max - ix_min)
    ih = max(0, iy_max - iy_min)

    intersection = iw * ih

    if intersection == 0:
        return 0.0

    area_a = (
        (ax_max - ax_min)
        * (ay_max - ay_min)
    )

    area_b = (
        (bx_max - bx_min)
        * (by_max - by_min)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def bbox_candidate_matrix(
    pred_boxes,
    gt_boxes,
    threshold=0.01
):

    pred_boxes = np.asarray(
        pred_boxes,
        dtype=np.float32
    )

    gt_boxes = np.asarray(
        gt_boxes,
        dtype=np.float32
    )

    # Prediction boxes
    px1 = pred_boxes[:, 0][:, None]
    px2 = pred_boxes[:, 1][:, None]
    py1 = pred_boxes[:, 2][:, None]
    py2 = pred_boxes[:, 3][:, None]

    # GT boxes
    gx1 = gt_boxes[:, 0][None, :]
    gx2 = gt_boxes[:, 1][None, :]
    gy1 = gt_boxes[:, 2][None, :]
    gy2 = gt_boxes[:, 3][None, :]

    ix1 = np.maximum(px1, gx1)
    ix2 = np.minimum(px2, gx2)

    iy1 = np.maximum(py1, gy1)
    iy2 = np.minimum(py2, gy2)

    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)

    intersection = iw * ih

    pred_area = (
        (px2 - px1)
        * (py2 - py1)
    )

    gt_area = (
        (gx2 - gx1)
        * (gy2 - gy1)
    )

    union = (
        pred_area
        + gt_area
        - intersection
    )

    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )

    return iou >= threshold


def match_predictions_to_gt_fast(
    predictions,
    gt_annotations,
    rle_cache,
    gt_rle_cache,
    iou_threshold=0.5,
    bbox_threshold=0.01
):
    """
    Fast prediction → GT matching.

    Uses bbox filtering before RLE IoU.
    """

    if not predictions or not gt_annotations:
        return []

    # ------------------------------------------------------
    # Select representative prediction
    # ------------------------------------------------------

    pred_annotations = []

    for pred in predictions:

        # Use s1 if available, otherwise s2, otherwise s3
        for seed in ["s1", "s2", "s3"]:

            if pred[seed] is not None:

                pred_annotations.append(
                    pred[seed]
                )

                break

    # ------------------------------------------------------
    # Bounding boxes
    # ------------------------------------------------------

    pred_boxes = [
        ann["bbox"]
        for ann in pred_annotations
    ]

    gt_boxes = [
        ann["bbox"]
        for ann in gt_annotations
    ]

    candidates = bbox_candidate_matrix(
        pred_boxes,
        gt_boxes,
        threshold=bbox_threshold
    )

    # ------------------------------------------------------
    # IoU matrix
    # ------------------------------------------------------

    iou_matrix = np.zeros(
        (
            len(pred_annotations),
            len(gt_annotations)
        ),
        dtype=np.float32
    )

    for i, pred_ann in enumerate(pred_annotations):

        pred_rle = rle_cache[
            pred_ann["anno_id"]
        ]

        candidate_indices = np.where(
            candidates[i]
        )[0]

        if len(candidate_indices) == 0:
            continue

        gt_rles = [
            gt_rle_cache[j]
            for j in candidate_indices
        ]

        ious = mask_utils.iou(
            [pred_rle],
            gt_rles,
            [0] * len(gt_rles)
            )[0]

        iou_matrix[
            i,
            candidate_indices
        ] = ious

    # ------------------------------------------------------
    # Hungarian matching
    # ------------------------------------------------------

    rows, cols = linear_sum_assignment(
        1.0 - iou_matrix
    )

    matches = []

    for i, j in zip(rows, cols):

        iou = iou_matrix[i, j]

        if iou >= iou_threshold:

            matches.append(
                (
                    i,
                    j,
                    float(iou)
                )
            )

    return matches


def calculate_pixel_entropy(
    masks
):
    """
    Mean pixel-wise predictive entropy.

    Only pixels in the union of the predicted
    masks are considered.
    """

    masks = [
        m for m in masks
        if m is not None
    ]

    if len(masks) < 2:
        return np.nan

    masks = np.asarray(
        masks,
        dtype=np.float32
    )

    p = masks.mean(axis=0)

    eps = 1e-8

    entropy = -(
        p * np.log(p + eps)
        +
        (1 - p) * np.log(
            1 - p + eps
        )
    )

    union = masks.max(axis=0) > 0

    if union.sum() == 0:
        return 0.0

    return entropy[union].mean()


def load_ground_truth(gt_path):

    with open(gt_path, "r") as f:
        gt = json.load(f)

    return gt


def organize_ground_truth(gt):

    gt_by_key = defaultdict(list)

    for ann in gt["annotations"]:

        if ann.get("iscrowd", 0):
            continue

        key = (
            ann["image_id"],
            ann["category_id"]
        )

        gt_by_key[key].append(ann)

    return gt_by_key


def get_image_sizes(gt):

    return {
        image["id"]: (
            image["height"],
            image["width"]
        )
        for image in gt["images"]
    }


def match_predictions_to_gt(
    predictions,
    gt_annotations,
    height,
    width,
    iou_threshold=0.5
):
    """
    Match ensemble consensus instances to GT masks.

    Returns:
        list of
        (prediction_index, gt_index, iou)
    """

    if not predictions or not gt_annotations:
        return []

    pred_masks = []

    for pred in predictions:

        # Select a representative prediction.
        #
        # For a 3-model consensus, use s1.
        # For cases where s1 is absent, use whichever
        # model is available.

        for seed in ["s1", "s2", "s3"]:

            if pred[seed] is not None:

                mask = polygon_to_mask(
                    pred[seed]["segmentation"],
                    height,
                    width
                )

                pred_masks.append(mask)
                break

    gt_masks = [
        polygon_to_mask(
            ann["segmentation"],
            height,
            width
        )
        for ann in gt_annotations
    ]

    iou_matrix = np.zeros(
        (len(pred_masks), len(gt_masks)),
        dtype=np.float32
    )

    for i, pred_mask in enumerate(pred_masks):

        for j, gt_mask in enumerate(gt_masks):

            iou_matrix[i, j] = mask_iou(
                pred_mask,
                gt_mask
            )

    rows, cols = linear_sum_assignment(
        1.0 - iou_matrix
    )

    matches = []

    for i, j in zip(rows, cols):

        iou = iou_matrix[i, j]

        if iou >= iou_threshold:

            matches.append(
                (
                    i,
                    j,
                    float(iou)
                )
            )

    return matches


def build_uncertainty_dataframe(
    matched_data,
    gt_by_key,
    image_sizes,
    gt_iou_threshold=0.5
):

    rows = []

    # Group matched predictions by image/category
    matched_by_key = defaultdict(list)

    for instance in matched_data:

        key = (
            instance["image_id"],
            instance["category_id"]
        )

        matched_by_key[key].append(instance)

    # ------------------------------------------------------
    # Process each image/category
    # ------------------------------------------------------

    for key, instances in matched_by_key.items():

        image_id, category_id = key

        height, width = image_sizes[image_id]

        gt_annotations = gt_by_key.get(
            key,
            []
        )

        # --------------------------------------------------
        # Match ensemble instances to GT
        # --------------------------------------------------

        matches = match_predictions_to_gt(
            instances,
            gt_annotations,
            height,
            width,
            iou_threshold=gt_iou_threshold
        )

        gt_match_dict = {
            pred_idx: (
                gt_idx,
                iou
            )
            for pred_idx, gt_idx, iou
            in matches
        }

        # --------------------------------------------------
        # Process every ensemble instance
        # --------------------------------------------------

        for idx, instance in enumerate(instances):

            num_models = instance["num_models"]

            # ----------------------------------------------
            # Pairwise uncertainty
            # ----------------------------------------------

            mean_iou, iou_std, disagreement, iou_s2_s3 = (
                calculate_mask_disagreement(
                    instance,
                    height,
                    width
                )
            )

            # ----------------------------------------------
            # Pixel entropy
            # ----------------------------------------------

            masks = []

            for seed in ["s1", "s2", "s3"]:

                if instance[seed] is not None:

                    masks.append(
                        polygon_to_mask(
                            instance[seed]["segmentation"],
                            height,
                            width
                        )
                    )

            entropy = calculate_pixel_entropy(
                masks
            )

            # ----------------------------------------------
            # Scores
            # ----------------------------------------------

            scores = [
                instance[seed]["score"]
                for seed in ["s1", "s2", "s3"]
                if instance[seed] is not None
            ]

            mean_score = (
                np.mean(scores)
                if scores
                else np.nan
            )

            score_std = (
                np.std(scores)
                if len(scores) > 1
                else np.nan
            )

            # ----------------------------------------------
            # Ground truth
            # ----------------------------------------------

            gt_iou = np.nan
            gt_error = np.nan

            if idx in gt_match_dict:

                _, gt_iou = gt_match_dict[idx]

                gt_error = 1.0 - gt_iou

            # ----------------------------------------------
            # Save row
            # ----------------------------------------------

            rows.append({

                "image_id": image_id,

                "category_id": category_id,

                "num_models": num_models,

                "iou_s1_s2": instance.get(
                    "iou_s1_s2",
                    np.nan
                ),

                "iou_s1_s3": instance.get(
                    "iou_s1_s3",
                    np.nan
                ),

                "iou_s2_s3": iou_s2_s3,

                "mean_pairwise_iou": mean_iou,

                "pairwise_iou_std": iou_std,

                "mask_disagreement": disagreement,

                "pixel_entropy": entropy,

                "mean_score": mean_score,

                "score_std": score_std,

                "gt_iou": gt_iou,

                "gt_error": gt_error
            })

    return pd.DataFrame(rows)



def build_uncertainty_dataframe_fast(
    matched_data,
    gt_by_key,
    image_sizes,
    gt_rle_cache,
    gt_iou_threshold=0.5,
    bbox_threshold=0.01
):

    # ------------------------------------------------------
    # Build prediction RLE cache
    # ------------------------------------------------------

    prediction_annotations = []

    for instance in matched_data:

        for seed in ["s1", "s2", "s3"]:

            if instance[seed] is not None:

                prediction_annotations.append(
                    instance[seed]
                )

    rle_cache = build_rle_cache(
        prediction_annotations,
        image_sizes
    )

    # ------------------------------------------------------
    # Group by image/category
    # ------------------------------------------------------

    matched_by_key = defaultdict(list)

    for instance in matched_data:

        key = (
            instance["image_id"],
            instance["category_id"]
        )

        matched_by_key[key].append(instance)

    rows = []

    # ------------------------------------------------------
    # Process image/category
    # ------------------------------------------------------

    for key, instances in matched_by_key.items():

        image_id, category_id = key

        gt_annotations = gt_by_key.get(
            key,
            []
        )

        # --------------------------------------------------
        # GT matching
        # --------------------------------------------------

        if gt_annotations:

            matches = match_predictions_to_gt_fast(
                instances,
                gt_annotations,
                rle_cache,
                gt_rle_cache,
                iou_threshold=gt_iou_threshold,
                bbox_threshold=bbox_threshold
            )

            gt_match_dict = {
                pred_idx: (
                    gt_idx,
                    iou
                )
                for pred_idx, gt_idx, iou in matches
            }

        else:

            gt_match_dict = {}

        # --------------------------------------------------
        # Process each instance
        # --------------------------------------------------

        for idx, instance in enumerate(instances):

            (
                mean_iou,
                iou_std,
                disagreement,
                iou_s2_s3
            ) = calculate_disagreement_fast(
                instance,
                rle_cache
            )

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            scores = [
                instance[seed]["score"]
                for seed in ["s1", "s2", "s3"]
                if instance[seed] is not None
            ]

            mean_score = (
                np.mean(scores)
                if scores
                else np.nan
            )

            score_std = (
                np.std(scores)
                if len(scores) > 1
                else np.nan
            )

            # ------------------------------------------------
            # GT error
            # ------------------------------------------------

            gt_iou = np.nan

            if idx in gt_match_dict:

                _, gt_iou = gt_match_dict[idx]

            gt_error = (
                1.0 - gt_iou
                if not np.isnan(gt_iou)
                else np.nan
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            rows.append({

                "image_id": image_id,

                "category_id": category_id,

                "num_models": instance["num_models"],

                "iou_s1_s2": instance.get(
                    "iou_s1_s2",
                    np.nan
                ),

                "iou_s1_s3": instance.get(
                    "iou_s1_s3",
                    np.nan
                ),

                "iou_s2_s3": iou_s2_s3,

                "mean_pairwise_iou": mean_iou,

                "pairwise_iou_std": iou_std,

                "mask_disagreement": disagreement,

                "mean_score": mean_score,

                "score_std": score_std,

                "gt_iou": gt_iou,

                "gt_error": gt_error
            })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    matched_dir =  Path("/data_hdd/jazibsdata/m2_uncertainty_quant/matched/")
    gt_path = Path("/data_hdd/jazibsdata/oam-tcd-coco-style-1024/coco_annotations_test.json")
    output_dir = matched_dir / "uncertainty_results"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    gt = load_ground_truth(gt_path)

    gt_by_key = organize_ground_truth(gt)

    image_sizes = get_image_sizes(gt)
    print("Building GT RLE cache...")
    gt_rle_cache = build_gt_rle_cache(
        gt["annotations"],
        image_sizes
        )
    print(
        f"Cached {len(gt_rle_cache)} GT masks."
    )
    matched_files = sorted(
        matched_dir.glob("matched_*.json")
    )

    print(
        f"\nFound {len(matched_files)} matched files."
    )

    all_results = []

    for file_idx, matched_path in enumerate(
        matched_files,
        start=1
    ):

        print("\n" + "=" * 70)

        print(
            f"[{file_idx}/{len(matched_files)}] "
            f"{matched_path.name}"
        )

        print("=" * 70)


        # ------------------------------------------------------
        # Load matched predictions
        # ------------------------------------------------------

        with open(matched_path, "r") as f:
            matched_data = json.load(f)

        print(
            f"Loaded {len(matched_data):,} "
            f"matched instances."
        )


        # ------------------------------------------------------
        # Build uncertainty dataframe
        # ------------------------------------------------------

        df = build_uncertainty_dataframe_fast(

            matched_data=matched_data,

            gt_by_key=gt_by_key,

            image_sizes=image_sizes,

            gt_rle_cache=gt_rle_cache,

            gt_iou_threshold=0.5,

            bbox_threshold=0.01
        )


        # ------------------------------------------------------
        # Extract configuration from filename
        #
        # matched_ConvNeXtS_LoRA_high.json
        #          ↓
        # backbone = ConvNeXtS
        # strategy = LoRA
        # datasplit = high
        # ------------------------------------------------------

        filename = matched_path.stem

        # Remove "matched_"
        config = filename.replace(
            "matched_",
            "",
            1
        )

        parts = config.split("_")

        if len(parts) >= 3:

            backbone = parts[0]
            strategy = parts[1]
            datasplit = parts[2]

        else:

            backbone = "unknown"
            strategy = "unknown"
            datasplit = "unknown"


        # ------------------------------------------------------
        # Add configuration information
        # ------------------------------------------------------

        df["backbone"] = backbone
        df["strategy"] = strategy
        df["datasplit"] = datasplit


        # ------------------------------------------------------
        # Save instance-level results
        # ------------------------------------------------------

        output_path = (
            output_dir
            / f"{config}_uncertainty.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )


        print(
            f"Saved: {output_path}"
        )


        # ------------------------------------------------------
        # Basic summary
        # ------------------------------------------------------

        print(
            f"Instances: {len(df):,}"
        )

        print(
            f"3-model matches: "
            f"{(df['num_models'] == 3).sum():,}"
        )

        print(
            f"2-model matches: "
            f"{(df['num_models'] == 2).sum():,}"
        )

        print(
            f"1-model matches: "
            f"{(df['num_models'] == 1).sum():,}"
        )


        # ------------------------------------------------------
        # Correlation between uncertainty and GT error
        # ------------------------------------------------------

        valid = df[
            df["mask_disagreement"].notna()
            &
            df["gt_error"].notna()
        ]

        if len(valid) >= 3:

            rho, p = spearmanr(
                valid["mask_disagreement"],
                valid["gt_error"]
            )

            print(
                f"Spearman uncertainty/error: "
                f"rho={rho:.3f}, "
                f"p={p:.4g}, "
                f"n={len(valid):,}"
            )

        else:

            print(
                "Not enough valid instances "
                "for correlation."
            )


        # ------------------------------------------------------
        # Store for combined results
        # ------------------------------------------------------

        all_results.append(df)


    # ==========================================================
    # Combine ALL configurations
    # ==========================================================

    if all_results:

        combined_df = pd.concat(
            all_results,
            ignore_index=True
        )

        combined_path = (
            output_dir
            / "all_configurations_uncertainty.csv"
        )

        combined_df.to_csv(
            combined_path,
            index=False
        )

        print("\n" + "=" * 70)

        print(
            f"Combined results saved to:\n"
            f"{combined_path}"
        )

        print(
            f"Total rows: "
            f"{len(combined_df):,}"
        )

        print("=" * 70)