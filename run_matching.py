##
import json
import re
from pathlib import Path
from collections import defaultdict
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_utils


SINGLE_TREE = 1
TREE_GROUP = 2
IM_H = 1024
IM_W = 1024


def load_predictions(json_path):
    with open(json_path, "r") as f:
        predictions = json.load(f)
    return predictions


def decode_segmentation(segmentation, height, width):
    """
    Decode a COCO polygon or RLE segmentation into a binary mask.
    """
    if isinstance(segmentation, list):
        # Polygon(s)
        rles = mask_utils.frPyObjects(
            segmentation,
            height,
            width
        )
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        # RLE
        rle = segmentation.copy()

        # JSON may store counts as a string
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
    else:
        raise TypeError(
            f"Unsupported segmentation type: {type(segmentation)}"
        )

    mask = mask_utils.decode(rle)
    # Multiple RLEs can produce H x W x N
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def polygon_to_rle(segmentation, height, width):
    """
    Convert COCO polygon segmentation to RLE.
    """
    rles = mask_utils.frPyObjects(
        segmentation,
        height,
        width
    )

    return mask_utils.merge(rles)


def polygon_mask_iou(seg_a, seg_b, height, width):
    rle_a = polygon_to_rle(seg_a, height, width)
    rle_b = polygon_to_rle(seg_b, height, width)
    return float(
        mask_utils.iou(
            [rle_a],
            [rle_b],
            [0]
        )[0, 0]
    )


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return intersection / union


def bbox_iou(box_a, box_b):
    """
    Bounding box format:
        [x_min, x_max, y_min, y_max]
    """

    ax_min, ax_max, ay_min, ay_max = box_a
    bx_min, bx_max, by_min, by_max = box_b

    # Intersection
    ix_min = max(ax_min, bx_min)
    ix_max = min(ax_max, bx_max)

    iy_min = max(ay_min, by_min)
    iy_max = min(ay_max, by_max)

    iw = max(0.0, ix_max - ix_min)
    ih = max(0.0, iy_max - iy_min)

    intersection = iw * ih

    if intersection == 0:
        return 0.0

    # Areas
    area_a = max(0.0, ax_max - ax_min) * max(0.0, ay_max - ay_min)
    area_b = max(0.0, bx_max - bx_min) * max(0.0, by_max - by_min)

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


def pairwise_mask_iou(masks_a, masks_b):
    """
    Returns matrix of shape:

        len(masks_a) x len(masks_b)
    """
    matrix = np.zeros((len(masks_a), len(masks_b)), dtype=np.float32)
    for i, mask_a in enumerate(masks_a):
        for j, mask_b in enumerate(masks_b):
            matrix[i, j] = mask_iou(mask_a, mask_b)
    return matrix


def hungarian_match(
    predictions_a,
    predictions_b,
    masks_a,
    masks_b,
    iou_threshold=0.5
):
    """
    Match predictions from model A to model B using mask IoU.

    Returns:
        matches:
            list of (idx_a, idx_b, iou)

        unmatched_a:
            indices in A without a match

        unmatched_b:
            indices in B without a match
    """
    if len(predictions_a) == 0 or len(predictions_b) == 0:
        return (
            [],
            list(range(len(predictions_a))),
            list(range(len(predictions_b)))
        )
    iou_matrix = pairwise_mask_iou(masks_a, masks_b)

    # Hungarian minimizes cost, so convert IoU to cost.
    cost_matrix = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    matched_a = set()
    matched_b = set()
    for i, j in zip(row_ind, col_ind):
        iou = iou_matrix[i, j]
        if iou >= iou_threshold:
            matches.append((i, j, float(iou)))
            matched_a.add(i)
            matched_b.add(j)
    unmatched_a = [
        i for i in range(len(predictions_a))
        if i not in matched_a
    ]
    unmatched_b = [
        j for j in range(len(predictions_b))
        if j not in matched_b
    ]
    return matches, unmatched_a, unmatched_b


def fast_mask_iou_matrix(
    preds_a,
    preds_b,
    height,
    width,
    bbox_threshold=0.01
):
    """
    Compute mask IoU only for spatially plausible pairs.
    """
    n_a = len(preds_a)
    n_b = len(preds_b)
    iou_matrix = np.zeros(
        (n_a, n_b),
        dtype=np.float32
    )

    for i, pred_a in enumerate(preds_a):
        for j, pred_b in enumerate(preds_b):
            # ------------------------------------------------
            # Cheap bbox filtering
            # ------------------------------------------------
            if bbox_iou(
                pred_a["bbox"],
                pred_b["bbox"]
            ) < bbox_threshold:

                continue
            # ------------------------------------------------
            # Expensive mask IoU
            # ------------------------------------------------
            iou_matrix[i, j] = polygon_mask_iou(
                pred_a["segmentation"],
                pred_b["segmentation"],
                height,
                width
            )
    return iou_matrix


def parse_filename(filename):
    """
    Parse:

        seed_backbone_strategy_datasplit.json

    Example:
        1_ConvNeXtS_LoRA_high.json

    Returns:
        {
            "seed": "1",
            "backbone": "ConvNeXtS",
            "strategy": "LoRA",
            "datasplit": "high"
        }
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 5:
        raise ValueError(
            f"Unexpected filename format: {filename}"
        )
    seed, backbone, strategy, datasplit, _ = parts
    return {
        "seed": seed,
        "backbone": backbone,
        "strategy": strategy,
        "datasplit": datasplit,
    }


def group_model_files(json_dir):
    """
    Group the 48 files into experimental conditions.

    Returns:

        {
            (backbone, strategy, datasplit):
                {
                    seed: filepath,
                    ...
                }
        }
    """
    json_dir = Path(json_dir)
    groups = defaultdict(dict)
    for path in json_dir.glob("*.json"):
        info = parse_filename(path.name)
        condition = (
            info["backbone"],
            info["strategy"],
            info["datasplit"]
        )
        groups[condition][info["seed"]] = path
    return groups


def split_by_category(predictions):
    """
    Split predictions into separate category lists.
    """
    by_category = defaultdict(list)
    for pred in predictions:
        by_category[pred["category_id"]].append(pred)
    return by_category


def match_two_models(
    predictions_a,
    predictions_b,
    iou_threshold=0.5
):
    """
    Match predictions from two models.

    Matching is performed:

        image
        ↓
        category
        ↓
        mask IoU
        ↓
        Hungarian assignment

    image_sizes:
        dict mapping image_id -> (height, width)
    """
    # Organize predictions by image
    preds_a_by_image = defaultdict(list)
    preds_b_by_image = defaultdict(list)

    for pred in predictions_a:
        preds_a_by_image[pred["image_id"]].append(pred)

    for pred in predictions_b:
        preds_b_by_image[pred["image_id"]].append(pred)

    all_image_ids = (
        set(preds_a_by_image.keys())
        | set(preds_b_by_image.keys())
    )
    results = []

    for image_id in all_image_ids:
        image_preds_a = preds_a_by_image.get(image_id, [])
        image_preds_b = preds_b_by_image.get(image_id, [])

        # Split by category
        categories = (
            set(p["category_id"] for p in image_preds_a)
            |
            set(p["category_id"] for p in image_preds_b)
        )

        height, width = IM_H, IM_W
        for category_id in categories:
            cat_a = [
                p for p in image_preds_a
                if p["category_id"] == category_id
            ]
            cat_b = [
                p for p in image_preds_b
                if p["category_id"] == category_id
            ]
            masks_a = [
                decode_segmentation(
                    p["segmentation"],
                    height,
                    width
                )
                for p in cat_a
            ]
            masks_b = [
                decode_segmentation(
                    p["segmentation"],
                    height,
                    width
                )
                for p in cat_b
            ]

            matches, unmatched_a, unmatched_b = hungarian_match(
                cat_a,
                cat_b,
                masks_a,
                masks_b,
                iou_threshold=iou_threshold
            )

            for idx_a, idx_b, iou in matches:
                results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "pred_a": cat_a[idx_a],
                    "pred_b": cat_b[idx_b],
                    "mask_iou": iou,
                    "matched": True
                })

            # Keep unmatched predictions
            for idx_a in unmatched_a:
                results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "pred_a": cat_a[idx_a],
                    "pred_b": None,
                    "mask_iou": None,
                    "matched": False
                })

            for idx_b in unmatched_b:
                results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "pred_a": None,
                    "pred_b": cat_b[idx_b],
                    "mask_iou": None,
                    "matched": False
                })
    return results


def match_three_models(
    predictions_by_seed,
    iou_threshold=0.5,
    reference_seed="s1"
):
    """
    Match three model seeds using seed 1 as reference.

    predictions_by_seed:
        {
            "1": [...],
            "2": [...],
            "3": [...]
        }

    Returns a list of consensus instances.
    """
    seeds = sorted(predictions_by_seed.keys())
    if reference_seed not in predictions_by_seed:
        raise ValueError(
            f"Reference seed {reference_seed} not found."
        )
    other_seeds = [
        s for s in seeds
        if s != reference_seed
    ]
    if len(seeds) != 3:
        raise ValueError(
            f"Expected 3 seeds, got {len(seeds)}"
        )

    # Organize each model by image/category
    grouped = {}
    for seed, predictions in predictions_by_seed.items():

        grouped[seed] = defaultdict(list)

        for pred in predictions:
            key = (
                pred["image_id"],
                pred["category_id"]
            )

            grouped[seed][key].append(pred)

    all_keys = set()
    for seed in seeds:
        all_keys.update(grouped[seed].keys())
    consensus_instances = []

    for image_id, category_id in all_keys:
        ref_preds = grouped[reference_seed].get(
            (image_id, category_id),
            []
        )

        height, width = IM_H, IM_W

        # Start with reference predictions
        instances = [
            {
                reference_seed: pred
            }
            for pred in ref_preds
        ]

        # Track predictions that are already matched
        matched_other_predictions = {
            seed: set()
            for seed in other_seeds
        }

        # --------------------------------------------------
        # Match each other seed against reference
        # --------------------------------------------------

        for seed in other_seeds:

            other_preds = grouped[seed].get(
                (image_id, category_id),
                []
            )

            ref_masks = [
                decode_segmentation(
                    inst[reference_seed]["segmentation"],
                    height,
                    width
                )
                for inst in instances
            ]

            other_masks = [
                decode_segmentation(
                    pred["segmentation"],
                    height,
                    width
                )
                for pred in other_preds
            ]

            if len(instances) > 0 and len(other_preds) > 0:
                iou_matrix = pairwise_mask_iou(
                    ref_masks,
                    other_masks
                )
                rows, cols = linear_sum_assignment(
                    1.0 - iou_matrix
                )

                for r, c in zip(rows, cols):
                    iou = iou_matrix[r, c]
                    if iou >= iou_threshold:
                        instances[r][seed] = other_preds[c]
                        matched_other_predictions[
                            seed
                        ].add(c)

        # --------------------------------------------------
        # Add unmatched predictions from other models
        # --------------------------------------------------
        for seed in other_seeds:
            other_preds = grouped[seed].get(
                (image_id, category_id),
                []
            )
            for idx, pred in enumerate(other_preds):
                if idx not in matched_other_predictions[seed]:
                    consensus_instances.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "members": {
                            seed: pred
                        },
                        "num_models": 1
                    })

        # --------------------------------------------------
        # Add reference-centered instances
        # --------------------------------------------------
        for instance in instances:
            consensus_instances.append({
                "image_id": image_id,
                "category_id": category_id,
                "members": instance,
                "num_models": len(instance)
            })
    return consensus_instances


def match_three_models_faster(
    preds_s1,
    preds_s2,
    preds_s3,
    height,
    width,
    mask_iou_threshold=0.5,
    bbox_threshold=0.01
):
    """
    Match predictions from three models/seeds using s1 as
    the reference model.

    Parameters
    ----------
    preds_s1, preds_s2, preds_s3 : list of dict
        Predictions for one image AND one category.

    height, width : int
        Image dimensions.

    mask_iou_threshold : float
        Minimum mask IoU required for a valid match.

    bbox_threshold : float
        Minimum bbox IoU required before calculating mask IoU.

    Returns
    -------
    consensus : list of dict

        Each element represents a consensus instance.

        Example:

        {
            "s1": prediction,
            "s2": prediction,
            "s3": prediction,
            "num_models": 3
        }

        If only s1 and s2 detect the tree:

        {
            "s1": prediction,
            "s2": prediction,
            "s3": None,
            "num_models": 2
        }

        If only s3 detects it:

        {
            "s1": None,
            "s2": None,
            "s3": prediction,
            "num_models": 1
        }
    """
    # ---------------------------------------------------------
    # Start with the s1 predictions as reference instances
    # ---------------------------------------------------------
    consensus = [
        {
            "s1": pred,
            "s2": None,
            "s3": None,
            "num_models": 1
        }
        for pred in preds_s1
    ]
    # ---------------------------------------------------------
    # Helper function to match one model to s1
    # ---------------------------------------------------------
    def match_to_reference(
        reference_preds,
        other_preds
    ):
        if not reference_preds:
            return [], list(range(len(other_preds)))
        if not other_preds:
            return [], []
        iou_matrix = fast_mask_iou_matrix(
            reference_preds,
            other_preds,
            height,
            width,
            bbox_threshold=bbox_threshold
        )
        rows, cols = linear_sum_assignment(
            1.0 - iou_matrix
        )
        matches = []
        matched_other = set()
        for i, j in zip(rows, cols):
            iou = iou_matrix[i, j]
            if iou >= mask_iou_threshold:
                matches.append(
                    (i, j, float(iou))
                )
                matched_other.add(j)
        unmatched_other = [
            j for j in range(len(other_preds))
            if j not in matched_other
        ]
        return matches, unmatched_other

    # ---------------------------------------------------------
    # Match s2 → s1
    # ---------------------------------------------------------
    matches_s2, unmatched_s2 = match_to_reference(
        preds_s1,
        preds_s2
    )

    for ref_idx, s2_idx, iou in matches_s2:
        consensus[ref_idx]["s2"] = preds_s2[s2_idx]
        consensus[ref_idx]["num_models"] += 1
        # Store pairwise agreement if useful later
        consensus[ref_idx]["iou_s1_s2"] = iou
    # ---------------------------------------------------------
    # Match s3 → s1
    # ---------------------------------------------------------
    matches_s3, unmatched_s3 = match_to_reference(
        preds_s1,
        preds_s3
    )

    for ref_idx, s3_idx, iou in matches_s3:
        consensus[ref_idx]["s3"] = preds_s3[s3_idx]
        consensus[ref_idx]["num_models"] += 1
        consensus[ref_idx]["iou_s1_s3"] = iou
    # ---------------------------------------------------------
    # Add s2-only predictions
    # ---------------------------------------------------------
    for idx in unmatched_s2:
        consensus.append({
            "s1": None,
            "s2": preds_s2[idx],
            "s3": None,
            "num_models": 1
        })
    # ---------------------------------------------------------
    # Add s3-only predictions
    # ---------------------------------------------------------
    for idx in unmatched_s3:
        consensus.append({
            "s1": None,
            "s2": None,
            "s3": preds_s3[idx],
            "num_models": 1
        })
    return consensus


if __name__ == "__main__":
    json_dir = Path("/data_hdd/jazibsdata/m2_uncertainty_quant/")
    output_dir = json_dir / "matched"
    output_dir.mkdir(exist_ok=True)
    groups = group_model_files(json_dir)
    print(f"Found {len(groups)} experimental conditions.")

    for condition, seed_files in groups.items():
        backbone, strategy, datasplit = condition
        print(
            f"\nProcessing: "
            f"{backbone} | {strategy} | {datasplit}"
        )
        print("Seeds:", seed_files.keys())
        # Make sure all three seeds exist
        required_seeds = {"s1", "s2", "s3"}
        if set(seed_files.keys()) != required_seeds:
            print(
                f"WARNING: expected seeds "
                f"{required_seeds}, "
                f"found {set(seed_files.keys())}"
            )
            continue
        # Load predictions
        predictions_by_seed = {}
        for seed in ["s1", "s2", "s3"]:
            path = seed_files[seed]
            print(
                f"  Loading {seed}: {path.name}"
            )
            with open(path, "r") as f:
                predictions_by_seed[seed] = json.load(f)

        # Match
        grouped = {
            seed: defaultdict(list)
            for seed in ["s1", "s2", "s3"]
        }
        for seed in ["s1", "s2", "s3"]:
            for pred in predictions_by_seed[seed]:
                key = (
                    pred["image_id"],
                    pred["category_id"]
                )
                grouped[seed][key].append(pred)

        all_keys = set()

        for seed in ["s1", "s2", "s3"]:
            all_keys.update(grouped[seed].keys())


        matched_results = []

        for image_id, category_id in all_keys:

            preds_s1 = grouped["s1"].get(
                (image_id, category_id),
                []
            )

            preds_s2 = grouped["s2"].get(
                (image_id, category_id),
                []
            )

            preds_s3 = grouped["s3"].get(
                (image_id, category_id),
                []
            )

            height, width = IM_H, IM_W

            consensus = match_three_models_faster(
                preds_s1=preds_s1,
                preds_s2=preds_s2,
                preds_s3=preds_s3,
                height=height,
                width=width,
                mask_iou_threshold=0.5,
                bbox_threshold=0.01
            )


            # Add image/category information
            for instance in consensus:

                instance["image_id"] = image_id
                instance["category_id"] = category_id

                matched_results.append(instance)
        ##
        #matched = match_three_models_faster(
        #    predictions_by_seed,
        #    iou_threshold=0.5,
        #    reference_seed="s1"
        #)
        #print(
        #    f"  Matched/collected instances: "
        #    f"{len(matched)}"
        #)
        # Save
        output_name = (
            f"matched_"
            f"{backbone}_"
            f"{strategy}_"
            f"{datasplit}.json"
        )
        output_path = output_dir / output_name
        with open(output_path, "w") as f:
            json.dump(
                matched_results,
                f,
                indent=2
            )