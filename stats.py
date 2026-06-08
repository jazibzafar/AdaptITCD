##
import os
import yaml
from utils import event_to_yml, yaml_to_dict_parser, write_dict_to_yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stats_yaml = "./statistics/combined_stats.yaml"
stats_dict = yaml_to_dict_parser(stats_yaml)

##

def extract_rows(data):
    rows = []
    for k, v in data.items():
        arch, strategy, split = k.split("_")
        rows.append({
            "name": k,
            "mask_ap": v["mask_map_50"][0],
            "bbox_ap": v["bbox_map_50"][0],
            "vram": v["peak_vram"][0],  # GB
            "time_hr": v["train_time"][0],
            "arch": arch,
            "strategy": strategy,
            "split": split
        })
    return rows

# ----------------------------
# Pareto frontier (sorted envelope)
# ----------------------------
def pareto_frontier_line(x, y):
    """
    Returns points on Pareto frontier sorted by x.
    Assumes:
      - minimize x
      - maximize y
    """
    idx = np.argsort(x)
    x_sorted = np.array(x)[idx]
    y_sorted = np.array(y)[idx]
    frontier_x = []
    frontier_y = []
    best_y = -np.inf
    for xi, yi in zip(x_sorted, y_sorted):
        if yi > best_y:
            frontier_x.append(xi)
            frontier_y.append(yi)
            best_y = yi
    return np.array(frontier_x), np.array(frontier_y)


def plot(ax, rows, cost_key, title, show_labels=True):

    strategies = sorted(set(r["strategy"] for r in rows))
    arches = sorted(set(r["arch"] for r in rows))

    markers = {
        a: m for a, m in zip(arches, ["o", "s", "^"])
    }  # ["o", "s", "^", "D", "X", "*"]
    colors = {"frozen": "#56B4E9", "full": "#F0E442", "lora": "#D55E00"}

    x = np.array([r[cost_key] for r in rows])
    y = np.array([r["mask_ap"] for r in rows])

    # Calculate a dynamic y-offset based on the fixed y-limits (30 to 50)
    # 3% of the total 20-point span keeps the text perfectly spaced below the marker
    y_offset = (50.0 - 30.0) * 0.03

    # ----------------------------
    # scatter points
    # ----------------------------
    for r in rows:
        ax.scatter(
            r[cost_key],
            r["mask_ap"],
            color=colors[r["strategy"]],
            marker=markers[r["arch"]],
            s=90,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        # _ar, _st, _sp = r["name"].split("_")
        _ar = r["arch"]
        _st = r["strategy"]
        _sp = r["split"]
        if _st == "full":
            text_name = _ar + " fft"
        else:
            text_name = _ar + " " + _st

        if show_labels:
            ax.text(
                r[cost_key],
                r["mask_ap"] - y_offset,  # Apply the offset to push text down
                text_name,  # r["name"].replace("s1_", "")
                fontsize=7,
                alpha=0.75,
                va="top",  # Align top of text box to coordinate
                ha="center",  # Center horizontally under marker
            )

    # ----------------------------
    # Pareto frontier line
    # ----------------------------
    fx, fy = pareto_frontier_line(x, y)
    ax.plot(fx, fy, linestyle="--", linewidth=2, color="black", alpha=0.5)

    # highlight frontier points
    # ax.scatter(fx, fy, color="black", s=120, zorder=5)

    # ----------------------------
    # legend proxies
    # ----------------------------
    for s in strategies:
        ax.scatter([], [], color=colors[s], label=s)

    for a in arches:
        ax.scatter([], [], marker=markers[a], color="gray", label=a)

    ax.legend(
        loc="lower center",  # Places it out of the way of your 30-50 AP data points
        frameon=True,  # Gives it a bounding box background
        fontsize=8,  # Matches the compact aesthetic of your text labels
        ncol=6,  # Arranges items in 2 columns to save vertical space
    )
    ax.set_title(title)
    ax.set_ylim(ymin=30.0, ymax=50.0)
    ax.set_xlabel(
        "Peak VRAM Usage (GB)" if cost_key == "vram" else "Training Time (hrs)"
    )
    ax.set_ylabel("Mask AP")
    ax.grid(True, alpha=0.3)


def plot_all(rows, cost):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    splits = ["quarter", "full"]
    # costs = ["vram", "time_hr"]

    for i, split in enumerate(splits):
        subset = [r for r in rows if r["split"] == split]
        plot(
            axes[i],
            subset,
            cost_key=cost,
            title=f"{'High Data' if split=='full' else 'Low Data'}"
        )

    plt.tight_layout()
    plt.show()


def plot_alt(rows, cost, split, f_name, disp):
    fig, ax = plt.subplots(figsize=(7, 6))
    subset = [r for r in rows if r["split"] == split]
    plot(
        ax,
        subset,
        cost_key=cost,
        title=f"{'High Data' if split == 'full' else 'Low Data'}/12GB VRAM Restriction"
    )
    plt.tight_layout()
    if disp:
        plt.show()
    else:
        plt.savefig(f_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_everything(rows, f_name, disp):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    splits = ["quarter", "full"]
    costs = ["time_hr", "vram"]

    m_split = {"quarter": "Low Data", "full": "High Data"}

    for i, split in enumerate(splits):
        subset = [r for r in rows if r["split"] == split]
        for j, cost in enumerate(costs):
            plot(
                axes[i, j],
                subset,
                cost_key=cost,
                title=f"{m_split[split].upper()} | {'VRAM' if cost=='vram' else 'Time'}",
                show_labels=True
            )

    plt.tight_layout()
    if disp:
        plt.show()
    else:
        plt.savefig(f_name, dpi=300, bbox_inches="tight")
        plt.close(fig)



rows = extract_rows(stats_dict)

restricted_rows = [row for row in rows if row["vram"]<=12.]


cost = "vram"  # costs = ["vram", "time_hr"]
split = "full"  # split  ["full", "quarter"]
f_name = f"./statistics/rest_vram_4x4.png"
# plot_alt(restricted_rows, cost, split, f_name, disp=True)
plot_everything(restricted_rows, f_name, False)











##
# ----------------------------
# Main plot function
# ----------------------------
# def plot(ax, rows, cost_key, title, show_labels=True):
#
#     strategies = sorted(set(r["strategy"] for r in rows))
#     arches = sorted(set(r["arch"] for r in rows))
#
#     markers = {a: m for a, m in zip(arches, ["o", "s", "^"])}  # ["o", "s", "^", "D", "X", "*"]
#     colors = {"frozen": "#56B4E9", "full": "#F0E442", "lora": "#D55E00"}
#
#     x = np.array([r[cost_key] for r in rows])
#     y = np.array([r["mask_ap"] for r in rows])
#
#     # ----------------------------
#     # scatter points
#     # ----------------------------
#     for r in rows:
#         ax.scatter(
#             r[cost_key],
#             r["mask_ap"],
#             color=colors[r["strategy"]],
#             marker=markers[r["arch"]],
#             s=90,
#             edgecolor="black",
#             linewidth=0.8,
#             alpha=0.9
#         )
#         # _ar, _st, _sp = r["name"].split("_")
#         _ar = r["arch"]
#         _st = r["strategy"]
#         _sp = r["split"]
#         if _st == "full":
#             text_name = _ar + " fft"
#         else:
#             text_name = _ar + " " + _st
#         if show_labels:
#             ax.text(
#                 r[cost_key],
#                 r["mask_ap"],
#                 text_name,  # r["name"].replace("s1_", "")
#                 fontsize=7,
#                 alpha=0.75
#             )
#
#     # ----------------------------
#     # Pareto frontier line
#     # ----------------------------
#     fx, fy = pareto_frontier_line(x, y)
#     ax.plot(fx, fy, linestyle="--", linewidth=2, color="black", alpha=0.5)
#
#     # highlight frontier points
#     # ax.scatter(fx, fy, color="black", s=120, zorder=5)
#
#     # ----------------------------
#     # legend proxies
#     # ----------------------------
#     for s in strategies:
#         ax.scatter([], [], color=colors[s], label=s)
#
#     for a in arches:
#         ax.scatter([], [], marker=markers[a], color="gray", label=a)
#
#     ax.set_title(title)
#     ax.set_ylim(ymin=30., ymax=50.)
#     ax.set_xlabel("Peak VRAM Usage (GB)" if cost_key == "vram" else "Training Time (hrs)")
#     ax.set_ylabel("Mask AP")
#     ax.grid(True, alpha=0.3)






# def plot_all(rows):
#
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#
#     splits = ["full", "quarter"]
#     costs = ["vram", "time_hr"]
#
#     for i, split in enumerate(splits):
#         subset = [r for r in rows if r["split"] == split]
#
#         for j, cost in enumerate(costs):
#             plot(
#                 axes[i, j],
#                 subset,
#                 cost_key=cost,
#                 title=f"{split.upper()} | {'VRAM' if cost=='vram' else 'Time'}"
#             )
#
#     plt.tight_layout()
#     plt.show()
##
# def is_pareto_efficient(y, x):
#     """
#     Maximize y (mask AP), minimize x (cost)
#     """
#     y = np.array(y)
#     x = np.array(x)
#
#     is_eff = np.ones(len(y), dtype=bool)
#
#     for i in range(len(y)):
#         if not is_eff[i]:
#             continue
#
#         dominated = (
#             (y >= y[i]) &
#             (x <= x[i]) &
#             ((y > y[i]) | (x < x[i]))
#         )
#         dominated[i] = False
#
#         if np.any(dominated):
#             is_eff[i] = False
#
#     return is_eff
#
#
# def extract_rows(data):
#     rows = []
#     for k, v in data.items():
#         rows.append({
#             "name": k,
#             "mask_ap": v["accuracy"]["mask_map_50"],
#             "bbox_ap": v["accuracy"]["bbox_map_50"],
#             "vram": v["efficiency"]["peak_vram_gb"] / 1024,  # GB
#             "time_hr": v["efficiency"]["train_time_sec"] / 3600,
#             "arch": v["arch"],
#             "strategy": v["strategy"],
#             "split": v["split"]
#         })
#     return rows
#
#
# def plot_pareto(rows, cost_key="vram", title="Pareto Frontier"):
#     strategies = sorted(set(r["strategy"] for r in rows))
#     arches = sorted(set(r["arch"] for r in rows))
#
#     markers = {a: m for a, m in zip(arches, ["o", "s", "^", "D", "X", "*"])}
#     # colors = {s: c for s, c in zip(strategies, ["tab:blue", "tab:orange", "tab:green", "tab:red"])}
#     colors = {"frozen": "#2563EB", "full": "#F59E0B", "lora": "#7C3AED"}
#     x = [r[cost_key] for r in rows]      # cost on X
#     y = [r["mask_ap"] for r in rows]     # AP on Y
#
#     pareto = is_pareto_efficient(y, x)
#
#     plt.figure(figsize=(9, 6))
#
#     for r, is_eff in zip(rows, pareto):
#         plt.scatter(
#             r[cost_key],
#             r["mask_ap"],
#             color=colors[r["strategy"]],
#             marker=markers[r["arch"]],
#             s=90,
#             alpha=0.6
#         )
#
#         if is_eff:
#             plt.scatter(
#                 r[cost_key],
#                 r["mask_ap"],
#                 color=colors[r["strategy"]],
#                 marker=markers[r["arch"]],
#                 s=200,
#                 edgecolor="black",
#                 linewidth=1.5
#             )
#
#     # legend proxies (strategy)
#     for s in strategies:
#         plt.scatter([], [], color=colors[s], label=s)
#
#     # legend proxies (arch)
#     for a in arches:
#         plt.scatter([], [], marker=markers[a], color="gray", label=a)
#
#     xlabel = "VRAM (GB)" if cost_key == "vram" else "Training Time (hrs)"
#     plt.xlabel(xlabel)
#     plt.ylabel("Mask AP (bbox_map_50)")
#     plt.title(title)
#     plt.grid(True, alpha=0.3)
#     plt.legend(loc="best", fontsize=9)
#
#     plt.show()
#
#
# # ----------------------------
# # main workflow
# # ----------------------------
# rows = extract_rows(stats_dict)
#
# splits = ["full", "quarter"]
#
# for split in splits:
#     subset = [r for r in rows if r["split"] == split]
#
#     plot_pareto(
#         subset,
#         cost_key="vram",
#         title=f"Pareto Frontier (Mask AP vs VRAM) - {split}"
#     )
#
#     plot_pareto(
#         subset,
#         cost_key="time_hr",
#         title=f"Pareto Frontier (Mask AP vs Training Time) - {split}"
#     )









##
# mappings_arch = {
#     "convnext": "DINOv3 ConvNeX Small",
#     "swin": "Satlas Swin Transformer Base",
#     "resnet50": "DeepForest ResNet-50"
# }
#
# mappings_split = {
#     "full": "High Data",
#     "quarter": "Low Data"
# }
#
# mappings_strat = {
#     "full": "FFT + LLRD",
#     "lora": "LoRA",
#     "frozen": "Frozen Backbone"
# }
#
# # Table A.1: 4-fold + Box
# rows = []
#
# for exp_name, exp_data in stats_dict.items():
#     rows.append({
#         "arch": exp_data["arch"],
#         "split": exp_data["split"],
#         "strategy": exp_data["strategy"],
#         "box_ap": exp_data["accuracy"]["bbox_map_50"],
#         "mask_ap": exp_data["accuracy"]["mask_map_50"],
#     })
#
# df = pd.DataFrame(rows)
#
# full_df = df[df["split"] == "full"]
# box_full = full_df.pivot_table(
#     index="strategy",
#     columns="arch",
#     values="box_ap",
#     aggfunc="mean"
# )
# mask_full = full_df.pivot_table(
#     index="strategy",
#     columns="arch",
#     values="mask_ap",
#     aggfunc="mean"
# )
#
# quarter_df = df[df["split"] == "quarter"]
# box_quarter = quarter_df.pivot_table(
#     index="strategy",
#     columns="arch",
#     values="box_ap",
#     aggfunc="mean"
# )
# mask_quarter = quarter_df.pivot_table(
#     index="strategy",
#     columns="arch",
#     values="mask_ap",
#     aggfunc="mean"
# )
#
#
# print("=== BOX AP (FULL) ===")
# box_full.to_latex("./statistics/box_map_fullfold.tex", index=True)
# print(box_full)
#
# print("\n=== MASK AP (FULL) ===")
# mask_full.to_latex("./statistics/mask_map_fullfold.tex", index=True)
# print(mask_full)
#
# print("\n=== BOX AP (QUARTER) ===")
# box_quarter.to_latex("./statistics/box_map_quarterfold.tex", index=True)
# print(box_quarter)
#
# print("\n=== MASK AP (QUARTER) ===")
# mask_quarter.to_latex("./statistics/mask_map_quarterfold.tex", index=True)
# print(mask_quarter)

##
