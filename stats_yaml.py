import os
from utils import event_to_yml, yaml_to_dict_parser, write_dict_to_yaml


# def collect_metrics(dir_root):
#     output_dict = {}
#     args_dict = yaml_to_dict_parser(os.path.join(dir_root, "args.yaml"))
#     key_mapping = {
#         'seed': 'seed',
#         'node': 'node',
#         'arch': 'arch_type',
#         'strategy': 'strategy',
#         'split': 'split'
#     }
#     output_dict.update({out_k: args_dict[in_k] for out_k, in_k in key_mapping.items()})
#     stats_dict = yaml_to_dict_parser(os.path.join(dir_root, "version_0", "stats.yaml"))
#     output_dict['last_epoch'] = int(stats_dict['epoch'])
#     accuracy_dict = yaml_to_dict_parser(os.path.join(dir_root, "accuracy_metrics.yaml"))
#     output_dict['accuracy'] = accuracy_dict
#     efficiency_dict = yaml_to_dict_parser(os.path.join(dir_root, "efficiency_metrics.yaml"))
#     output_dict['efficiency'] = efficiency_dict
#     return output_dict
#
#
# root = "/mnt/cluster/data_hdd/jazibmodels/Fine_Tuning_Strategies/"
# # test_root = "/mnt/cluster/data_hdd/jazibmodels/Fine_Tuning_Strategies/s1_resnet50_full_quarter/"
# list_dir = os.listdir(root)
# # list_dir = [dir for dir in list_dir if dir.startswith("s1") and "half" not in dir]
# list_dir = [dir for dir in list_dir if dir.startswith("s2") or dir.startswith("s3")]
##
#  create event from yaml
# for dir in list_dir:
#     event_loc = os.path.join(root, dir, "version_0")
#     event_to_yml(event_loc)

# leftover = os.path.join(root, "s1_resnet50_full_quarter", "version_1")
# event_to_yml(leftover)

##
# list_dir = [dir for dir in list_dir if dir.startswith("s3")]
# metrics = {}
# write_metrics = "./statistics/all_seed_3_stats.yaml"
#
# for dir in list_dir:
#     path = os.path.join(root, dir)
#     metrics[dir] = collect_metrics(path)
#
# write_dict_to_yaml(write_metrics, metrics)
##
# SINGLE METRIC FILE
import yaml
from collections import defaultdict
import numpy as np


files = ["./statistics/all_seed_1_stats.yaml",
        "./statistics/all_seed_2_stats.yaml",
        "./statistics/all_seed_3_stats.yaml"]


scale = {
    "bbox_map_50": 100,
    "mask_map_50": 100,
    "peak_vram_gb": 1 / 1024,         # MB -> GB
    "train_time_sec": 1 / 3600,  # sec -> hr
}


# exp_name -> metric -> list of values
metrics = defaultdict(lambda: defaultdict(list))

for file in files:
    with open(file, "r") as f:
        data = yaml.safe_load(f)

    for key, stats in data.items():
        # Remove seed prefix: s1_convnext_frozen_full -> convnext_frozen_full
        exp_name = "_".join(key.split("_")[1:])

        metrics[exp_name]["bbox_map_50"].append(
            stats["accuracy"]["bbox_map_50"] * scale["bbox_map_50"]
        )
        metrics[exp_name]["mask_map_50"].append(
            stats["accuracy"]["mask_map_50"] * scale["mask_map_50"]
        )
        metrics[exp_name]["peak_vram"].append(
            stats["efficiency"]["peak_vram_gb"] * scale["peak_vram_gb"]
        )
        metrics[exp_name]["train_time"].append(
            stats["efficiency"]["train_time_sec"] * scale["train_time_sec"]
        )

combined_metrics = {}

for exp_name, exp_metrics in metrics.items():
    combined_metrics[exp_name] = {
        metric: [
            float(np.mean(values)),
            float(np.std(values, ddof=0)),  # use ddof=1 for sample std
        ]
        for metric, values in exp_metrics.items()
    }

print(combined_metrics)
##
write_dict_to_yaml("statistics/combined_stats.yaml", combined_metrics)