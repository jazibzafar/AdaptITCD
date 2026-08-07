##
import torch
import os
from dataset import OAMTCDCOCODataset
from SimplifiedMaskRCNN import LitMaskRCNN
import yaml


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# TODO: Get a list of all the checkpoints
model_root = "/home/jazib/projects/ft_uq_bbs/"
data_root = "/home/jazib/projects/data/oam-tcd-coco-style-1024/"
list_ckpt = [
    's1_convnext_full_full',
    's1_convnext_lora_full'
]
current_ckpt = list_ckpt[1]



# TODO: Path augmentation for each checkpoint os.path.join(root, ckpt[i])
ckpt_path = os.path.join(model_root, current_ckpt, "last.ckpt")

with open(os.path.join(model_root, current_ckpt, "args.yaml"), 'r') as stream:
    args = yaml.safe_load(stream)

args = DotDict(args)


##

# [X] seed, backbone, strat, split separation
seed, backbone, strategy, split = current_ckpt.split("_")

from modules import ViTDetBackbone, TorchvisionSwinV2Backbone, ResNet50Backbone, BackboneWithFPN, ConvNeXtBackbone



split_dict = {
    "full": {"train": [0, 1, 2, 3], "val": [4]},
    "half": {"train": [0, 1], "val": [4]},
    "quarter": {"train": [1], "val": [4]}
}


backbone = ConvNeXtBackbone(
            backbone_type='small',
            checkpoint_path=None,
            apply_lora=True if args.strategy == "lora" else False,
            lora_rank=args.lora_rank
        )
backbone_with_fpn = BackboneWithFPN(body=backbone)


# DONE: Load the dataset
train_dataset = OAMTCDCOCODataset(
    root_dir=data_root,
    folds=split_dict[args.split]["train"],
    transforms=None,
    return_masks=True
)

val_dataset = OAMTCDCOCODataset(
    root_dir=data_root,
    folds=split_dict[args.split]["val"],
    transforms=None,
    return_masks=True)


test_dataset = OAMTCDCOCODataset(
    root_dir=data_root,
    split='test',
    folds=None,
    return_masks=True
)

# TODO: Load the lightning model
model = LitMaskRCNN.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    backbone=backbone_with_fpn,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    args=args
)


# TODO: Define the test step

##
# DONE: Sensible output test [PASSED]
# with torch.no_grad():
#     model.eval()
#     dddd = model.model(torch.rand(1, 3, 1024, 1024).to("cuda"))

# DONE: Actual output test [PASSED]
# import matplotlib.pyplot as plt
#
#
# sample = test_dataset[333]
# img, label = sample
#
# with torch.no_grad():
#     model.eval()
#     out = model.model(img.unsqueeze(0).to("cuda"))
#
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(img.permute(1,2,0).numpy())
# ax[0].axis('off')
# ax[1].imshow(label['masks'].squeeze(0).numpy())
# ax[1].axis('off')
# plt.show()

# from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
#
# def show_out(img_tensor, pred):
#     img_uint8 = (img_tensor.cpu().clamp(0, 1) * 255).byte()
#     boxes = pred["boxes"]
#     masks = pred["masks"].squeeze(1) > 0.5 if "masks" in pred else None
#     result_img = img_uint8
#     if boxes.numel() > 0:
#         result_img = draw_bounding_boxes(result_img, boxes, colors="red", width=3)
#         if masks is not None and masks.any():
#             result_img = draw_segmentation_masks(result_img, masks, alpha=0.5, colors="blue")
#     plt.imshow(result_img.permute(1,2,0))
#     plt.axis("off")
#     plt.show()
#
#
# show_out(img, out[0])
