import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

# Input CSV
INPUT_CSV = Path(
    "/data_hdd/jazibsdata/m2_uncertainty_quant/matched/uncertainty_results/all_configurations_uncertainty.csv"
)

# Output directory
OUTPUT_DIR = Path(
    "/data_hdd/jazibsdata/m2_uncertainty_quant/matched/uncertainty_results/metrics/"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Category labels
# ------------------------------------------------------------

# CHANGE THESE IF YOUR CATEGORY IDs ARE DIFFERENT
CATEGORY_NAMES = {
    1: "single-tree",
    2: "tree-group"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_spearman(
    x,
    y
):
    """
    Calculate Spearman correlation while handling NaNs.
    """

    valid = (
        pd.notna(x)
        &
        pd.notna(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, np.nan, len(x)

    rho, p = spearmanr(
        x,
        y
    )

    return rho, p, len(x)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print(f"Input: {INPUT_CSV}")

df = pd.read_csv(
    INPUT_CSV
)

print(
    f"Loaded {len(df):,} rows"
)

print(
    f"Columns:\n{list(df.columns)}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_id",
    "category_id",
    "num_models",
    "mask_disagreement",
    "mean_pairwise_iou",
    "gt_iou",
    "gt_error",
    "mean_score",
    "score_std",
    "backbone",
    "strategy",
    "datasplit"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# CATEGORY NAMES
# ============================================================

df["category_name"] = (
    df["category_id"]
    .map(CATEGORY_NAMES)
    .fillna(
        df["category_id"].astype(str)
    )
)


# ============================================================
# 2. BASIC OVERALL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)


overall = pd.DataFrame({
    "metric": [
        "Number of instances",
        "Mean mask disagreement",
        "Median mask disagreement",
        "Mean pairwise IoU",
        "Mean GT IoU",
        "Median GT IoU",
        "Mean GT error",
        "Median GT error",
        "Mean model score",
        "Mean score std"
    ],

    "value": [

        len(df),

        df["mask_disagreement"].mean(),

        df["mask_disagreement"].median(),

        df["mean_pairwise_iou"].mean(),

        df["gt_iou"].mean(),

        df["gt_iou"].median(),

        df["gt_error"].mean(),

        df["gt_error"].median(),

        df["mean_score"].mean(),

        df["score_std"].mean()
    ]
})

print(overall.to_string(index=False))

overall.to_csv(
    OUTPUT_DIR / "overall_summary.csv",
    index=False
)


# ============================================================
# 3. CONSENSUS / MODEL AGREEMENT
# ============================================================

print("\n")
print("=" * 70)
print("MODEL AGREEMENT")
print("=" * 70)


consensus = (
    df["num_models"]
    .value_counts()
    .sort_index()
    .rename_axis("num_models")
    .reset_index(name="n")
)

consensus["percentage"] = (
    consensus["n"]
    / len(df)
    * 100
)

print(
    consensus.to_string(index=False)
)

consensus.to_csv(
    OUTPUT_DIR / "model_agreement.csv",
    index=False
)


# ============================================================
# 4. SINGLE-TREE VS TREE-GROUP
# ============================================================

print("\n")
print("=" * 70)
print("SINGLE-TREE VS TREE-GROUP")
print("=" * 70)


category_summary = (
    df
    .groupby(
        [
            "category_id",
            "category_name"
        ]
    )
    .agg(

        n=(
            "image_id",
            "size"
        ),

        mean_uncertainty=(
            "mask_disagreement",
            "mean"
        ),

        median_uncertainty=(
            "mask_disagreement",
            "median"
        ),

        mean_pairwise_iou=(
            "mean_pairwise_iou",
            "mean"
        ),

        mean_gt_iou=(
            "gt_iou",
            "mean"
        ),

        median_gt_iou=(
            "gt_iou",
            "median"
        ),

        mean_gt_error=(
            "gt_error",
            "mean"
        ),

        mean_score=(
            "mean_score",
            "mean"
        ),

        mean_score_std=(
            "score_std",
            "mean"
        )
    )
    .reset_index()
)

print(
    category_summary.to_string(index=False)
)

category_summary.to_csv(
    OUTPUT_DIR / "category_summary.csv",
    index=False
)


# ============================================================
# 5. UNCERTAINTY → ERROR CORRELATION
# ============================================================

print("\n")
print("=" * 70)
print("UNCERTAINTY → ERROR CORRELATION")
print("=" * 70)


rho, p, n = safe_spearman(
    df["mask_disagreement"],
    df["gt_error"]
)

print(
    f"Overall:\n"
    f"  Spearman rho = {rho:.4f}\n"
    f"  p-value      = {p:.4g}\n"
    f"  n            = {n:,}"
)


# ------------------------------------------------------------
# By category
# ------------------------------------------------------------

category_correlations = []

for (
    category_id,
    category_name
), subset in df.groupby(
    [
        "category_id",
        "category_name"
    ]
):

    rho, p, n = safe_spearman(
        subset["mask_disagreement"],
        subset["gt_error"]
    )

    category_correlations.append({

        "category_id": category_id,

        "category_name": category_name,

        "n": n,

        "spearman_rho": rho,

        "p_value": p
    })


category_correlations = pd.DataFrame(
    category_correlations
)

print("\nBy category:")

print(
    category_correlations.to_string(
        index=False
    )
)

category_correlations.to_csv(
    OUTPUT_DIR
    / "category_uncertainty_error_correlation.csv",
    index=False
)


# ============================================================
# 6. CONFIGURATION SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("CONFIGURATION SUMMARY")
print("=" * 70)


config_summary = (
    df
    .groupby(
        [
            "backbone",
            "strategy",
            "datasplit"
        ]
    )
    .agg(

        n=(
            "image_id",
            "size"
        ),

        mean_uncertainty=(
            "mask_disagreement",
            "mean"
        ),

        median_uncertainty=(
            "mask_disagreement",
            "median"
        ),

        mean_pairwise_iou=(
            "mean_pairwise_iou",
            "mean"
        ),

        mean_gt_iou=(
            "gt_iou",
            "mean"
        ),

        median_gt_iou=(
            "gt_iou",
            "median"
        ),

        mean_gt_error=(
            "gt_error",
            "mean"
        ),

        mean_score=(
            "mean_score",
            "mean"
        ),

        mean_score_std=(
            "score_std",
            "mean"
        ),

        three_model_fraction=(
            "num_models",
            lambda x: (x == 3).mean()
        )
    )
    .reset_index()
)

config_summary[
    "three_model_percentage"
] = (
    config_summary[
        "three_model_fraction"
    ]
    * 100
)

config_summary = config_summary.drop(
    columns="three_model_fraction"
)

print(
    config_summary.to_string(index=False)
)

config_summary.to_csv(
    OUTPUT_DIR / "configuration_summary.csv",
    index=False
)


# ============================================================
# 7. CONFIGURATION-SPECIFIC UNCERTAINTY/ERROR CORRELATION
# ============================================================

print("\n")
print("=" * 70)
print("CONFIGURATION-SPECIFIC CORRELATIONS")
print("=" * 70)


config_correlations = []

for (
    backbone,
    strategy,
    datasplit
), subset in df.groupby(
    [
        "backbone",
        "strategy",
        "datasplit"
    ]
):

    rho, p, n = safe_spearman(
        subset["mask_disagreement"],
        subset["gt_error"]
    )

    config_correlations.append({

        "backbone": backbone,

        "strategy": strategy,

        "datasplit": datasplit,

        "n": n,

        "spearman_rho": rho,

        "p_value": p
    })


config_correlations = pd.DataFrame(
    config_correlations
)

print(
    config_correlations.to_string(
        index=False
    )
)

config_correlations.to_csv(
    OUTPUT_DIR
    / "configuration_uncertainty_error_correlation.csv",
    index=False
)


# ============================================================
# 8. CONFIGURATION + CATEGORY
# ============================================================

print("\n")
print("=" * 70)
print("CONFIGURATION × CATEGORY")
print("=" * 70)


config_category_summary = (
    df
    .groupby(
        [
            "backbone",
            "strategy",
            "datasplit",
            "category_id",
            "category_name"
        ]
    )
    .agg(

        n=(
            "image_id",
            "size"
        ),

        mean_uncertainty=(
            "mask_disagreement",
            "mean"
        ),

        median_uncertainty=(
            "mask_disagreement",
            "median"
        ),

        mean_gt_iou=(
            "gt_iou",
            "mean"
        ),

        mean_gt_error=(
            "gt_error",
            "mean"
        ),

        mean_score=(
            "mean_score",
            "mean"
        )
    )
    .reset_index()
)

print(
    config_category_summary.to_string(
        index=False
    )
)

config_category_summary.to_csv(
    OUTPUT_DIR
    / "configuration_category_summary.csv",
    index=False
)


# ============================================================
# 9. UNCERTAINTY CALIBRATION
# ============================================================

print("\n")
print("=" * 70)
print("UNCERTAINTY CALIBRATION")
print("=" * 70)


valid = df[
    df["mask_disagreement"].notna()
    &
    df["gt_error"].notna()
].copy()


# Ten equally populated uncertainty bins
valid["uncertainty_bin"] = pd.qcut(
    valid["mask_disagreement"],
    q=10,
    duplicates="drop"
)


calibration = (
    valid
    .groupby(
        "uncertainty_bin",
        observed=True
    )
    .agg(

        n=(
            "image_id",
            "size"
        ),

        mean_uncertainty=(
            "mask_disagreement",
            "mean"
        ),

        mean_gt_error=(
            "gt_error",
            "mean"
        ),

        mean_gt_iou=(
            "gt_iou",
            "mean"
        ),

        mean_score=(
            "mean_score",
            "mean"
        )
    )
    .reset_index()
)

print(
    calibration.to_string(index=False)
)

calibration.to_csv(
    OUTPUT_DIR / "uncertainty_calibration.csv",
    index=False
)


# ============================================================
# 10. RISK-COVERAGE
# ============================================================

print("\n")
print("=" * 70)
print("RISK-COVERAGE")
print("=" * 70)


valid = valid.sort_values(
    "mask_disagreement"
)


coverage_levels = np.arange(
    0.1,
    1.01,
    0.1
)


risk_coverage_rows = []


for coverage in coverage_levels:

    n = max(
        1,
        int(
            len(valid)
            * coverage
        )
    )

    subset = valid.iloc[:n]

    risk_coverage_rows.append({

        "coverage": coverage,

        "n": n,

        "mean_uncertainty": (
            subset["mask_disagreement"]
            .mean()
        ),

        "mean_gt_error": (
            subset["gt_error"]
            .mean()
        ),

        "mean_gt_iou": (
            subset["gt_iou"]
            .mean()
        )
    })


risk_coverage = pd.DataFrame(
    risk_coverage_rows
)

print(
    risk_coverage.to_string(
        index=False
    )
)

risk_coverage.to_csv(
    OUTPUT_DIR / "risk_coverage.csv",
    index=False
)


# ============================================================
# 11. MODEL SCORE VS UNCERTAINTY
# ============================================================

print("\n")
print("=" * 70)
print("CONFIDENCE VS UNCERTAINTY")
print("=" * 70)


rho, p, n = safe_spearman(
    df["mean_score"],
    df["mask_disagreement"]
)

print(
    f"Score vs mask disagreement:\n"
    f"  Spearman rho = {rho:.4f}\n"
    f"  p-value      = {p:.4g}\n"
    f"  n            = {n:,}"
)


# ============================================================
# 12. HIGH-CONFIDENCE / HIGH-UNCERTAINTY CASES
# ============================================================

print("\n")
print("=" * 70)
print("HIGH-CONFIDENCE / HIGH-UNCERTAINTY")
print("=" * 70)


score_threshold = (
    df["mean_score"]
    .quantile(0.75)
)

uncertainty_threshold = (
    df["mask_disagreement"]
    .quantile(0.75)
)


high_conf_high_uncertainty = df[
    (df["mean_score"] >= score_threshold)
    &
    (
        df["mask_disagreement"]
        >= uncertainty_threshold
    )
].copy()


print(
    f"Mean score threshold: "
    f"{score_threshold:.4f}"
)

print(
    f"Uncertainty threshold: "
    f"{uncertainty_threshold:.4f}"
)

print(
    f"Number of cases: "
    f"{len(high_conf_high_uncertainty):,}"
)


high_conf_high_uncertainty.to_csv(
    OUTPUT_DIR
    / "high_confidence_high_uncertainty.csv",
    index=False
)


# ============================================================
# 13. TOP 20 MOST UNCERTAIN INSTANCES
# ============================================================

print("\n")
print("=" * 70)
print("MOST UNCERTAIN INSTANCES")
print("=" * 70)


top_uncertain = (
    df
    .sort_values(
        "mask_disagreement",
        ascending=False
    )
    .head(20)
)


print(
    top_uncertain[
        [
            "image_id",
            "category_name",
            "backbone",
            "strategy",
            "datasplit",
            "num_models",
            "mask_disagreement",
            "mean_score",
            "gt_iou",
            "gt_error"
        ]
    ].to_string(index=False)
)


top_uncertain.to_csv(
    OUTPUT_DIR
    / "top_20_uncertain_instances.csv",
    index=False
)


# ============================================================
# 14. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Results saved to:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print("\nFiles created:")

for path in sorted(
    OUTPUT_DIR.glob("*.csv")
):

    print(
        f"  {path.name}"
    )