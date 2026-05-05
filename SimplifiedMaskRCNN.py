import numpy as np
import math
import os
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import lightning as L
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from modules import ViTDetBackbone, TorchvisionSwinV2Backbone, ResNet50Backbone, BackboneWithFPN
from modules import AltTorchvisionResNet50Backbone, AltTorchvisionSwinV2Backbone
from dataset import OAMTCDCOCODataset, get_train_transforms
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.callbacks.finetuning import BaseFinetuning
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import write_png
from tifffile import imread
import torch.nn as nn
import torchvision.transforms.functional as F
import argparse
import sys

# TODO: Move seed_everything to main.
# TODO: Add seed options as args in dict form {s1: 1111, s2: 6147, s3: 4319}
L.seed_everything(1234)


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def collate_fn(batch):
    return tuple(zip(*batch))


