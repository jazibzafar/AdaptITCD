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
# SINGLE EXPERIMENT METRICS FILE
s1_met = yaml_to_dict_parser("./statistics/all_seed_1_stats.yaml")
s2_met = yaml_to_dict_parser("./statistics/all_seed_2_stats.yaml")
s3_met = yaml_to_dict_parser("./statistics/all_seed_3_stats.yaml")


##
import re
from collections import defaultdict
import numpy as np


metrics_to_keep = {
    "bbox_map_50": 100,
    "mask_map_50": 100,
    "peak_vram_gb": 1 / 1024,         # MB -> GB
    "train_time_sec": 1 / 3600,  # sec -> hr
}

combined_dict = {}

exp_dicts = [s1_met, s2_met, s3_met]
exp_dicts = [
    {
        re.sub(f"^s\d+_", "", key): value
        for key, value in d.items()
    }
    for d in exp_dicts
]

for exp_name in exp_dicts[0]:

    metric_values = defaultdict(list)

    for seed_dict in exp_dicts:
        exp_data = seed_dict[exp_name]

        for category_metrics in exp_data.values():
            for metric_name, value in category_metrics.items():

                # Ignore unwanted metrics
                if metric_name not in ["accuracy", "efficiency"]:
                    continue
                elif metric_name == "accuracy":
                    metric_values["accuracy"][metric_name].append(
                        value * metrics_to_keep[metric_name]
                    )
                else:
                    metric_values["efficiency"][metric_name].append(
                        value * metrics_to_keep[metric_name]
                    )

    combined_dict[exp_name] = {
        metric_name: [
            float(np.mean(values)),
            float(np.std(values, ddof=1)),
        ]
        for metric_name, values in metric_values.items()
    }

