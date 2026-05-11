##
# This file is temporarily added to the repo. Remove for later.
import torch
from modules import ResNet50Backbone, ConvNeXtBackbone, TorchvisionSwinV2Backbone


def no_weight_decay(name, param):
    if param.ndim == 1:
        return True
    if name.endswith(".bias"):
        return True
    keywords = [
        "norm",
        "bn",
        "relative_position",
        "relative_coords_table",
        "logit_scale",
    ]
    return any(k in name.lower() for k in keywords)


def get_resnet_layer_id(name):
    if name.startswith("conv1") or name.startswith("bn1"):
        return 0
    elif name.startswith("layer1"):
        return 1
    elif name.startswith("layer2"):
        return 2
    elif name.startswith("layer3"):
        return 3
    elif name.startswith("layer4"):
        return 4
    else:
        return 5


def get_convnext_layer_id(name):
    # stem
    if name.startswith("downsample_layers.0"):
        return 0
    # stage 0
    elif name.startswith("stages.0"):
        return 1
    # stage 1
    elif name.startswith("stages.1"):
        return 2
    # stage 2
    elif name.startswith("stages.2"):
        return 3
    # stage 3
    elif name.startswith("stages.3"):
        return 4
    # classifier / norms
    else:
        return 5


def get_swin_layer_id(name):
    # patch embedding
    if name.startswith("features.0"):
        return 0
    # stage 1
    elif name.startswith("features.1"):
        return 1
    # patch merge
    elif name.startswith("features.2"):
        return 1
    # stage 2
    elif name.startswith("features.3"):
        return 2
    elif name.startswith("features.4"):
        return 2
    # stage 3
    elif name.startswith("features.5"):
        return 3
    elif name.startswith("features.6"):
        return 3
    # stage 4
    elif name.startswith("features.7"):
        return 4
    else:
        return 5


def build_llrd_groups(
    model,
    model_type,
    base_lr=1e-4,
    weight_decay=0.05,
    layer_decay=0.8,
):
    if model_type == "resnet50":
        get_layer_id = get_resnet_layer_id
    elif model_type == "convnext":
        get_layer_id = get_convnext_layer_id
    elif model_type == "swin":
        get_layer_id = get_swin_layer_id
    else:
        raise ValueError(model_type)

    num_layers = 6
    lr_scales = {
        i: layer_decay ** (num_layers - 1 - i)
        for i in range(num_layers)
    }

    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = get_layer_id(name)
        decay_type = (
            "no_decay"
            if no_weight_decay(name, param)
            else "decay"
        )
        group_name = f"{layer_id}_{decay_type}"

        if group_name not in param_groups:
            param_groups[group_name] = {
                "params": [],
                "lr": base_lr * lr_scales[layer_id],
                "weight_decay": (
                    0.0 if decay_type == "no_decay"
                    else weight_decay
                ),
            }
        param_groups[group_name]["params"].append(param)
    return list(param_groups.values())


##

model = ConvNeXtBackbone()

body = model.body
output = build_llrd_groups(body, 'convnext')

##
# state_dict = model.body.state_dict()
# state_dict_keys = list(state_dict.keys())
# write_path = './swin_keys.txt'
#
# with open(write_path, 'w') as f:
#     for key in state_dict_keys:
#         f.write(key + '\n')

