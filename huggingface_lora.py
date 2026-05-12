##
import torch
import transformers
import peft

print(torch.__version__)
print(transformers.__version__)
print(peft.__version__)



##
from peft import LoraConfig, get_peft_model
from dinov3_convnext import ConvNeXt
from modules import ConvNeXtBackbone

# convnext = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],)
convnext = ConvNeXtBackbone()

convnext_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["pwconv1", "pwconv2"], # Must match your print-out above
    lora_dropout=0.05,
    bias="none"
)

convnext_peft = get_peft_model(convnext, convnext_config)
convnext_peft.print_trainable_parameters()
##
from modules import ResNet50Backbone, TorchvisionSwinV2Backbone

r50 = ResNet50Backbone()

r50_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["conv1", "conv2", "conv3"],
    lora_dropout=0.05,
    bias="none",
    # modules_to_save=["fc"]
)

r50_peft = get_peft_model(r50, r50_config)
r50_peft.print_trainable_parameters()
##
swin = TorchvisionSwinV2Backbone()  # renamed AltTorchvisionSwinV2Backbone

swin_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # In Torchvision Swin V2, these are the most effective targets
    target_modules=["qkv", "proj"],
    lora_dropout=0.05,
    bias="none",
    # modules_to_save=["head"]
)

swin_peft = get_peft_model(swin, swin_config)
swin_peft.print_trainable_parameters()
##
from modules import BackboneWithFPN
from utils import count_params


fpn = BackboneWithFPN(convnext_peft)
print(count_params(fpn.fpn))