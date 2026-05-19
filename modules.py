import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from lora_pytorch import LoRA
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models import swin_v2_b
from torchvision.models import resnet50
from utils import model_replace_prefix, count_params
from dinov3_convnext import ConvNeXt
from peft import LoraConfig, get_peft_model


def freeze_except_lora(model):
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_norm(norm, num_channels):
    if norm == "LN":
        return nn.GroupNorm(1, num_channels)
    elif norm == "BN":
        return nn.BatchNorm2d(num_channels)
    elif norm == "":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported norm: {norm}")


class SimpleFeaturePyramid(nn.Module):
    """
    Pure PyTorch implementation of ViTDet SimpleFeaturePyramid.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        norm="LN",
        top_block: nn.Module = None,
    ):
        super().__init__()

        self.scale_factors = scale_factors
        self.top_block = top_block
        self.stages = nn.ModuleList()

        use_bias = norm == ""

        for scale in scale_factors:
            layers = []
            cur_channels = in_channels

            if scale == 4.0:
                layers += [
                    nn.ConvTranspose2d(cur_channels, cur_channels // 2, 2, 2),
                    get_norm(norm, cur_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(cur_channels // 2, cur_channels // 4, 2, 2),
                ]
                cur_channels = cur_channels // 4

            elif scale == 2.0:
                layers += [
                    nn.ConvTranspose2d(cur_channels, cur_channels // 2, 2, 2),
                ]
                cur_channels = cur_channels // 2

            elif scale == 1.0:
                pass

            elif scale == 0.5:
                layers += [
                    nn.MaxPool2d(2, 2),
                ]

            else:
                raise NotImplementedError(f"Unsupported scale factor {scale}")

            layers += [
                nn.Conv2d(cur_channels, out_channels, 1, bias=use_bias),
                get_norm(norm, out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=use_bias),
                get_norm(norm, out_channels),
            ]

            self.stages.append(nn.Sequential(*layers))

    def forward(self, x):
        results = []

        for stage in self.stages:
            results.append(stage(x))

        if self.top_block is not None:
            results.extend(self.top_block(results[-1]))

        # outputs = {
        #     f"p{idx + 2}": feat for idx, feat in enumerate(results)
        # }
        outputs = {
            str(idx): feat for idx, feat in enumerate(results)
        }
        return outputs


class ViTBackbone(nn.Module):
    """
    ViT backbone with:
      - optional LoRA
      - custom checkpoint loading
      - automatic pos_embed resizing for different img_size
      - compatible with features_only=True timm wrappers
    """
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        img_size=1024,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size

        # Create ViT with features_only
        self.vit = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            img_size=img_size,
        )

        # Save embed dim
        self.embed_dim = self.vit.feature_info.channels()[0]

    def _resize_pos_embed(self, pos_embed_ckpt, vit_module):
        """
        Resize positional embeddings from checkpoint to match current img_size
        """
        cls_tok_ckpt = pos_embed_ckpt[:, :1, :]
        patch_tok_ckpt = pos_embed_ckpt[:, 1:, :]

        old_size = int(patch_tok_ckpt.shape[1] ** 0.5)
        new_size = int(vit_module.patch_embed.num_patches ** 0.5)

        patch_tok_ckpt = patch_tok_ckpt.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        patch_tok_resized = F.interpolate(
            patch_tok_ckpt, size=(new_size, new_size), mode="bicubic", align_corners=False  # mode="bilinear"
        )
        patch_tok_resized = patch_tok_resized.permute(0, 2, 3, 1).reshape(1, new_size*new_size, -1)

        return torch.cat([cls_tok_ckpt, patch_tok_resized], dim=1)

    def load_checkpoint(self, ckpt_path):
        """
        Load custom ViT checkpoint (handles arbitrary image sizes and LoRA wrapping)
        """
        state_dict = torch.load(ckpt_path, map_location="cpu")['teacher']
        # state_dict = {"model."+k: v for k, v in state_dict.items()}

        # vit_module = self._get_vit_module()

        # Resize pos_embed if needed
        if "pos_embed" in state_dict:
            if state_dict["pos_embed"].shape != self.vit.model.pos_embed.shape:
                state_dict["pos_embed"] = self._resize_pos_embed(state_dict["pos_embed"], self.vit.model)

        # Load weights into the actual ViT module
        missing, unexpected = self.vit.model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    def apply_lora(self, lora_rank):
        # Apply LoRA if requested
        self.vit = LoRA.from_module(self.vit, rank=lora_rank)
        freeze_except_lora(self.vit)
        print("LoRA wrapping applied and non-LoRA params frozen")
        print(f"trainable params: {count_params(self.vit)}")

    def freeze_parameters(self):
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.vit(x)[0]


class ViTDetBackbone(nn.Module):
    """
    ViT backbone + Simple FPN
    """
    def __init__(
        self,
        img_size=1024,
        out_channels=256,
    ):
        super().__init__()
        self.vit = ViTBackbone(
            model_name="vit_base_patch16_224",
            img_size=img_size,
        )
        self.fpn = SimpleFeaturePyramid(
            in_channels=self.vit.embed_dim,
            out_channels=out_channels,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.vit(x)
        return self.fpn(x)


class ResNet50Backbone(nn.Module):
    def __init__(self, checkpoint_path=None, apply_lora=False, lora_rank=16):
        super().__init__()
        # 1. Load the base ResNet50
        # weights=None if loading a full custom state_dict
        base_model = resnet50(weights=None)

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            state_dict = model_replace_prefix(state_dict, "", "backbone.body.")
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(f"Custom weights loaded from {checkpoint_path}")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        # ResNet stages: layer1=1/4, layer2=1/8, layer3=1/16, layer4=1/32
        return_nodes = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        # ResNet50 out_channels for these layers are [256, 512, 1024, 2048]
        self.body = create_feature_extractor(base_model, return_nodes=return_nodes)
        self.out_channels = [256, 512, 1024, 2048]

        if apply_lora:
            lora_config = LoraConfig(
               r=lora_rank, lora_alpha=32, target_modules=["conv1", "conv2", "conv3"], lora_dropout=0.05, bias="none",
            )
            self.body = get_peft_model(self.body, lora_config)
            self.body.gradient_checkpointing_enable()
            print("LoRA applied successfully")
            self.body.print_trainable_parameters()

    def freeze_parameters(self):
        for param in self.body.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.body(x)


class TorchvisionSwinV2Backbone(nn.Module):
    def __init__(self, checkpoint_path=None, apply_lora=False, lora_rank=16):
        super().__init__()

        base_model = swin_v2_b(weights=None)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            state_dict = model_replace_prefix(state_dict, "", "backbone.backbone.")
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(f"Custom weights loaded from {checkpoint_path}")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        if apply_lora:
            lora_config = LoraConfig(
               r=lora_rank, lora_alpha=32, target_modules=["qkv", "proj"], lora_dropout=0.05, bias="none",
            )
            base_model = get_peft_model(base_model, lora_config)
            base_model.gradient_checkpointing_enable()
            print("LoRA applied successfully")
            base_model.print_trainable_parameters()

        self.body = base_model
        # We only need the hierarchical features
        self.features = self.body.features
        # Channels for FPN
        self.out_channels = [128, 256, 512, 1024]

    def freeze_parameters(self):
        for param in self.body.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features[0](x)
        x = self.features[1](x)
        out0 = x
        x = self.features[2](x)
        x = self.features[3](x)
        out1 = x
        x = self.features[4](x)
        x = self.features[5](x)
        out2 = x
        x = self.features[6](x)
        x = self.features[7](x)
        out3 = x
        return {
            "0": out0.permute(0, 3, 1, 2),
            "1": out1.permute(0, 3, 1, 2),
            "2": out2.permute(0, 3, 1, 2),
            "3": out3.permute(0, 3, 1, 2),
        }


class ConvNeXtBackbone(nn.Module):
    def __init__(self, backbone_type='small', checkpoint_path=None, apply_lora=False, lora_rank=16):
        super().__init__()
        self.backbone_type = backbone_type
        self.body_params = self.get_depth_and_dims(self.backbone_type)
        base_model = ConvNeXt(**self.body_params)
        self.depths = self.body_params['depths']
        self.out_channels = self.body_params['dims']
        self.feat_names = ['0', '1', '2', '3']
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(f"ConvNext-{self.backbone_type} weights loaded from {checkpoint_path}")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        if apply_lora:
            lora_config = LoraConfig(
               r=lora_rank, lora_alpha=32, target_modules=["pwconv1", "pwconv2"], lora_dropout=0.05, bias="none",
            )
            base_model = get_peft_model(base_model, lora_config)
            base_model.gradient_checkpointing_enable()
            print("LoRA applied successfully")
            base_model.print_trainable_parameters()
        self.body = base_model

    @staticmethod
    def get_depth_and_dims(backbone_type):
        convnext_sizes = {"tiny": dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
                          "small": dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
                          "base": dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
                          "large": dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])}
        return convnext_sizes[backbone_type]

    def freeze_parameters(self):
        for param in self.body.parameters():
            param.requires_grad = False

    def forward(self, x):
        output_tuple = self.body.get_intermediate_layers(x, n=4, reshape=True)
        return {k: v for k, v in zip(self.feat_names, output_tuple)}


class BackboneWithFPN(nn.Module):
    def __init__(self, body, out_channels=256):
        super().__init__()
        self.body = body
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=body.out_channels,
            out_channels=out_channels,
            # extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
