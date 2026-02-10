import numpy as np
import torch
from collections import OrderedDict


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