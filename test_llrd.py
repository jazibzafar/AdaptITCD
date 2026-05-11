##
# This file is temporarily added to the repo. Remove for later.
import torch
from modules import ResNet50Backbone, ConvNeXtBackbone, TorchvisionSwinV2Backbone


# def no_weight_decay(name, param):
#     if param.ndim == 1:
#         return True
#     if name.endswith(".bias"):
#         return True
#     keywords = [
#         "norm", "bn", "relative_position", "relative_coords_table", "logit_scale",
#     ]
#     return any(k in name.lower() for k in keywords)

def no_weight_decay(name, param):
    # 1D params (biases and norm scales) never get weight decay
    if param.ndim <= 1:
        return True
    # Consolidate keywords; name.lower() handled once
    keywords = {
        "norm", "bn", "ln", "gn",  # Normalization
        "pos_embed", "position_embedding",  # Positional Info
        "rel_pos", "relative_position",  # Swin/ViT variants
        "cls_token", "mask_token", "dist_token",  # Special tokens
        "logit_scale"  # CLIP/Timm models
    }
    name = name.lower()
    return any(k in name for k in keywords)

def get_layer_id(arch_type, name):
    map_dict = {
        'resnet50': {"conv1": 0, "bn1": 0, "layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4},
        'convnext': {"downsample_layers.0": 0, "stages.0": 1, "stages.1": 2, "stages.2": 3, "stages.3": 4},
        'swin': {"features.0": 0, "features.1": 1, "features.2": 1, "features.3": 2, "features.4": 2,
                 "features.5": 3, "features.6": 3, "features.7": 4}
    }
    mapping = map_dict[arch_type]
    return next((val for key, val in mapping.items() if key in name), 5)


def build_llrd_groups(model, backbone_type, base_lr=1e-4, weight_decay=0.05, layer_decay=0.75,):
    num_layers = 6
    lr_scales = {
        i: layer_decay ** (num_layers - 1 - i)
        for i in range(num_layers)
    }

    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = get_layer_id(backbone_type, name)
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

model_conv = ResNet50Backbone()

body = model_conv.body
output = build_llrd_groups(body, 'resnet50')


##
from modules import BackboneWithFPN
from LitMaskRCNN import LitMaskRCNN

backbone = BackboneWithFPN(model_conv)

model = LitMaskRCNN.build_mask_rcnn(backbone, 3, img_size=224)
model_output = build_llrd_groups(model, 'resnet50')


##
# def get_resnet_layer_id(name):
#     mapping = {"conv1": 0, "bn1": 0, "layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
#     if name in list(mapping.keys()):
#         return mapping[name]
#     else:
#         return 5
#
#
# def get_convnext_layer_id(name):
#     mapping = {
#         "downsample_layers.0": 0, "stages.0": 1, "stages.1": 2, "stages.2": 3, "stages.3": 4
#     }
#     if name in list(mapping.keys()):
#         return mapping[name]
#     else:
#         return 5
#
#
# def get_swin_layer_id(name):
#     mapping = {
#         "features.0": 0,
#         "features.1": 1, "features.2": 1,
#         "features.3": 2, "features.4": 2,
#         "features.5": 3, "features.6": 3,
#         "features.7": 4
#     }
#     if name in list(mapping.keys()):
#         return mapping[name]
#     else:
#         return 5


# def get_resnet_layer_id(name):
#     if name.startswith("conv1") or name.startswith("bn1"):
#         return 0
#     elif name.startswith("layer1"):
#         return 1
#     elif name.startswith("layer2"):
#         return 2
#     elif name.startswith("layer3"):
#         return 3
#     elif name.startswith("layer4"):
#         return 4
#     else:
#         return 5


# def get_convnext_layer_id(name):
#     # stem
#     if name.startswith("downsample_layers.0"):
#         return 0
#     # stage 0
#     elif name.startswith("stages.0"):
#         return 1
#     # stage 1
#     elif name.startswith("stages.1"):
#         return 2
#     # stage 2
#     elif name.startswith("stages.2"):
#         return 3
#     # stage 3
#     elif name.startswith("stages.3"):
#         return 4
#     # classifier / norms
#     else:
#         return 5


# def get_swin_layer_id(name):
#     # patch embedding
#     if name.startswith("features.0"):
#         return 0
#     # stage 1
#     elif name.startswith("features.1"):
#         return 1
#     # patch merge
#     elif name.startswith("features.2"):
#         return 1
#     # stage 2
#     elif name.startswith("features.3"):
#         return 2
#     elif name.startswith("features.4"):
#         return 2
#     # stage 3
#     elif name.startswith("features.5"):
#         return 3
#     elif name.startswith("features.6"):
#         return 3
#     # stage 4
#     elif name.startswith("features.7"):
#         return 4
#     else:
#         return 5


##
# state_dict = model.body.state_dict()
# state_dict_keys = list(state_dict.keys())
# write_path = './swin_keys.txt'
#
# with open(write_path, 'w') as f:
#     for key in state_dict_keys:
#         f.write(key + '\n')

