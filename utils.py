import numpy as np
import torch
from collections import OrderedDict
import yaml
import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse


def replace_prefix(text, prefix_add, prefix_rem):
    if text.startswith(prefix_rem):
        return prefix_add + text[len(prefix_rem):]
    return text


def model_replace_prefix(in_state_dict, prefix_add, prefix_rem):
    pairings = [
        (src_key, replace_prefix(src_key, prefix_add, prefix_rem))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return
    out_state_dict = {}
    for src_key, dest_key in pairings:
        # print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    return OrderedDict(out_state_dict)


def count_params(model):
    """Count trainable parameters of a PyTorch Model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    return nb_params


def yaml_to_dict_parser(file):
    with open(file, 'r') as f:
        yml_params = yaml.safe_load(f)
    return yml_params


def write_dict_to_yaml(yaml_file, dict_in, default_flow_style=False):
    with open(yaml_file, 'w') as outfile:
        yaml.dump(dict_in, outfile, default_flow_style=default_flow_style)


def write_dict_to_csv(file_to_write, dict_to_write):
    with open(file_to_write, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_to_write.items():
            writer.writerow([key, value])


def acc_event_to_dict(path_event, dict_in):
    ea = EventAccumulator(path_event).Reload()
    tags = ea.Tags()['scalars']

    for tag in tags:
        tag_values = []
        for event in ea.Scalars(tag):
            tag_values.append(event.value)
            # the conditional below ensures that singletons are not interpreted as lists.
            if len(tag_values) > 1:
                dict_in[tag] = tag_values
            else:
                dict_in[tag] = event.value
    return dict_in


def event_to_yml(path):
    ignore = "hparams.yaml"
    stat_dict = {}
    for dname in os.listdir(path):
        if not dname == ignore:
            path_event = os.path.join(path, dname)
            acc_event_to_dict(path_event, stat_dict)
    write_file = os.path.join(path, "stats.yaml")
    write_dict_to_yaml(write_file, stat_dict)
