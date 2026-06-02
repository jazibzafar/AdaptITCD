##
import os
import yaml
from utils import event_to_yml, yaml_to_dict_parser, write_dict_to_yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


combined_stats = yaml_to_dict_parser("./statistics/combined_stats.yaml")

##
print(f"{combined_stats['convnext_frozen_full']['bbox_map_50'][0]:.1f} ± {combined_stats['convnext_frozen_full']['bbox_map_50'][1]:.2f}")
##
# Creating DataFrame for Tables 1 - 4
rows = []
# ' ± '
for name, stats in combined_stats.items():
    arch, strategy, split = name.split("_")
    rows.append({
        "arch": arch,
        "strategy": strategy,
        "split": split,
        "bbox_ap": f"{stats['bbox_map_50'][0]:0.2f} ± {stats['bbox_map_50'][1]:0.2f}",
        "mask_ap": f"{stats['mask_map_50'][0]:0.2f} ± {stats['mask_map_50'][1]:0.2f}"
    })
df = pd.DataFrame(rows)

high_df = df[df["split"] == "full"]
low_df = df[df["split"] == "quarter"]

##
# Table 1
low_box = low_df.pivot_table(
    index="strategy",
    columns="arch",
    values="bbox_ap",
    aggfunc="first"
)
# Table 2
low_mask = low_df.pivot_table(
    index="strategy",
    columns="arch",
    values="mask_ap",
    aggfunc="first"
)
# Table 3
high_box = high_df.pivot_table(
    index="strategy",
    columns="arch",
    values="bbox_ap",
    aggfunc="first"
)
# Table 4
high_mask = high_df.pivot_table(
    index="strategy",
    columns="arch",
    values="mask_ap",
    aggfunc="first"
)


low_box.to_latex("./statistics/low_box.tex")
low_mask.to_latex("./statistics/low_mask.tex")
high_box.to_latex("./statistics/high_box.tex")
high_mask.to_latex("./statistics/high_mask.tex")
##
# Preparing for DataFrames 5 - 8
fro_dict = {key: value for key, value in combined_stats.items() if "frozen" in key}
unf_dict = {key: value for key, value in combined_stats.items() if "frozen" not in key}

##
# Creating DataFrames for 5 -8
rows = []
for name, stats in unf_dict.items():
    arch, strategy, split = name.split("_")
    fro_dict_key = f"{arch}_frozen_{split}"
    bbox_ap = stats['bbox_map_50'][0] - fro_dict[fro_dict_key]['bbox_map_50'][0]
    mask_ap = stats['mask_map_50'][0] - fro_dict[fro_dict_key]['mask_map_50'][0]
    rows.append({
        "arch": arch,
        "strategy": strategy,
        "split": split,
        "bbox_ap": f"{bbox_ap:0.2f}",
        "mask_ap": f"{mask_ap:0.2f}"
    })

gain_df = pd.DataFrame(rows)

gain_high = gain_df[gain_df["split"] == "full"]
gain_low = gain_df[gain_df["split"] == "quarter"]
##
# Table 5
print("Table 5")
gain_low_box = gain_low.pivot_table(
    index="strategy",
    columns="arch",
    values="bbox_ap",
    aggfunc="first"
)
print(gain_low_box)
gain_low_box.to_latex("./statistics/gain_low_box.tex")
# Table 6
print("Table 6")
gain_low_mask = gain_low.pivot_table(
    index="strategy",
    columns="arch",
    values="mask_ap",
    aggfunc="first"
)
print(gain_low_mask)
gain_low_mask.to_latex("./statistics/gain_low_mask.tex")
# Table 7
print("Table 7")
gain_high_box = gain_high.pivot_table(
    index="strategy",
    columns="arch",
    values="bbox_ap",
    aggfunc="first"
)
print(gain_high_box)
gain_high_box.to_latex("./statistics/gain_high_box.tex")
# Table 8
print("Table 8")
gain_high_mask = gain_high.pivot_table(
    index="strategy",
    columns="arch",
    values="mask_ap",
    aggfunc="first"
)
print(gain_high_mask)
gain_high_mask.to_latex("./statistics/gain_high_mask.tex")



