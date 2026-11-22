# # ##
# OLD OPTIMIZER
# def configure_optimizers(self):
#     params = [p for p in self.model.parameters() if p.requires_grad]
#
#     decay_params = []
#     no_decay_params = []
#     for n, p in self.named_parameters():
#         if not p.requires_grad:
#             continue
#         if p.ndim < 2 or "bias" in n or "norm" in n or "pos_embed" in n:
#             no_decay_params.append(p)
#         else:
#             decay_params.append(p)
#
#     optimizer = torch.optim.AdamW([
#         {'params': decay_params, 'weight_decay': self.args.weight_decay},
#         {'params': no_decay_params, 'weight_decay': 0.0}
#     ], lr=self.args.lr)
#
#     total_steps = self.trainer.estimated_stepping_batches
#     warmup_steps = self.args.warmup_steps
#
#     def lr_lambda(current_step):
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))
#
#         progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#         return 0.5 * (1.0 + math.cos(math.pi * progress))
#
#     scheduler = LambdaLR(optimizer, lr_lambda)
#
#     return {
#         "optimizer": optimizer,
#         "lr_scheduler": {
#             "scheduler": scheduler,
#             "interval": "step",
#             "frequency": 1,
#         },
#     }
# # ##
# # # ANOTHER OPTIMIZER BACKUP
# # def configure_optimizers(self):
# #     lr = self.args.lr
# #     t1, t2, t3 = self.get_tier_mapper()
# #
# #     if self.args.arch_type == 'vit':
# #         body_params = set(id(p) for p in self.model.backbone.vit.vit.parameters())
# #     else:
# #         body_params = set(id(p) for p in self.model.backbone.body.parameters())
# #     # Remember Neck/FPN params do not change and are part of head params
# #     head_params = [p for p in self.model.parameters() if id(p) not in body_params and p.requires_grad]
# #
# #     # Strategy 1: Adaptive Learning Rates
# #     if getattr(self.args, "strategy", None) == "adaptive":
# #         param_groups = [
# #             {'params': t1, 'lr': lr * 0.25, 'weight_decay': self.args.weight_decay},
# #             {'params': t2, 'lr': lr * 0.50, 'weight_decay': self.args.weight_decay},
# #             {'params': t3, 'lr': lr * 0.75, 'weight_decay': self.args.weight_decay},
# #             {'params': head_params, 'lr': lr, 'weight_decay': self.args.weight_decay}
# #         ]
# #     else:
# #         # Default or Gradual (uses your bias/norm exclusion logic)
# #         if getattr(self.args, "strategy", None) == "gradual":
# #             param_groups = [
# #                 {"params": t1, "lr": 0.0, "weight_decay": self.args.weight_decay},
# #                 {"params": t2, "lr": 0.0, "weight_decay": self.args.weight_decay},
# #                 {"params": t3, "lr": 0.0, "weight_decay": self.args.weight_decay},
# #                 {"params": head_params, "lr": lr, "weight_decay": self.args.weight_decay},
# #             ]
# #
# #         decay_params = []
# #         no_decay_params = []
# #         for n, p in self.named_parameters():
# #             if not p.requires_grad: continue
# #             if p.ndim < 2 or "bias" in n or "norm" in n or "pos_embed" in n:
# #                 no_decay_params.append(p)
# #             else:
# #                 decay_params.append(p)
# #         param_groups = [
# #             {'params': decay_params, 'weight_decay': self.args.weight_decay},
# #             {'params': no_decay_params, 'weight_decay': 0.0}
# #         ]
# #
# #     optimizer = torch.optim.AdamW(param_groups, lr=lr)
# #
# #     total_steps = self.trainer.estimated_stepping_batches
# #     warmup_steps = self.args.warmup_steps
# #
# #     def lr_lambda(current_step):
# #         if current_step < warmup_steps:
# #             return float(current_step) / float(max(1, warmup_steps))
# #         progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
# #         return 0.5 * (1.0 + math.cos(math.pi * progress))
# #
# #     scheduler = LambdaLR(optimizer, lr_lambda)
# #     return {
# #         "optimizer": optimizer,
# #         "lr_scheduler": {
# #             "scheduler": scheduler,
# #             "interval": "step",
# #         },
# #     }
# ##
# def configure_optimizers(self):
#     lr = self.args.lr
#     wd = self.args.weight_decay
#     strategy = getattr(self.args, "strategy", None)
#
#     t1, t2, t3 = self.get_tier_mapper()
#
#     # --- HEAD PARAMS ---
#     if self.args.arch_type == "vit":
#         body_params = set(id(p) for p in self.model.backbone.vit.vit.parameters())
#     else:
#         body_params = set(id(p) for p in self.model.backbone.body.parameters())
#
#     head_params = [p for p in self.model.parameters() if id(p) not in body_params]
#
#     param_groups = []
#     # STRATEGY: ADAPTIVE
#     if strategy == "adaptive":
#         for tier, tier_lr in zip([t1, t2, t3], [lr * 0.25, lr * 0.50, lr * 0.75]):
#             decay, no_decay = self.split_decay(tier)
#             param_groups.append({"params": decay, "lr": tier_lr, "weight_decay": wd})
#             param_groups.append({"params": no_decay, "lr": tier_lr, "weight_decay": 0.0})
#         # Head params
#         decay, no_decay = self.split_decay(head_params)
#         param_groups.append({"params": decay, "lr": lr, "weight_decay": wd})
#         param_groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
#     # STRATEGY: GRADUAL
#     elif strategy == "gradual":
#         for tier in [t1, t2, t3]:
#             decay, no_decay = self.split_decay(tier)
#             param_groups.append({"params": decay, "lr": 0.0, "weight_decay": wd})
#             param_groups.append({"params": no_decay, "lr": 0.0, "weight_decay": 0.0})
#         # Head params train immediately
#         decay, no_decay = self.split_decay(head_params)
#         param_groups.append({"params": decay, "lr": lr, "weight_decay": wd})
#         param_groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
#     # DEFAULT STRATEGY
#     else:
#         decay = []
#         no_decay = []
#
#         for n, p in self.named_parameters():
#             if not p.requires_grad: continue
#             if p.ndim < 2 or "bias" in n or "norm" in n or "pos_embed" in n:
#                 no_decay.append(p)
#             else:
#                 decay.append(p)
#
#         param_groups = [
#             {"params": decay, "weight_decay": wd},
#             {"params": no_decay, "weight_decay": 0.0},
#         ]
#
#     optimizer = torch.optim.AdamW(param_groups, lr=lr)
#
#     # Scheduler
#     total_steps = self.trainer.estimated_stepping_batches
#     warmup_steps = self.args.warmup_steps
#
#     def lr_lambda(current_step):
#         if current_step < warmup_steps:
#             return float(current_step) / float(max(1, warmup_steps))
#         progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
#         return 0.5 * (1.0 + math.cos(math.pi * progress))
#
#     scheduler = LambdaLR(optimizer, lr_lambda)
#
#     return {
#         "optimizer": optimizer,
#         "lr_scheduler": {
#             "scheduler": scheduler,
#             "interval": "step",
#         },
#     }

# DEPRECATED SWIN V2 BACKBONE
# class TorchvisionSwinV2Backbone(nn.Module):
#     def __init__(self, checkpoint_path=None, apply_lora=False, lora_rank=4):
#         super().__init__()
#         # 1. Initialize the base model
#         # We use weights=None if you are loading a full custom state_dict
#         base_model = swin_v2_b(weights=None)
#
#         if checkpoint_path:
#             state_dict = torch.load(checkpoint_path, map_location='cpu')
#             state_dict = model_replace_prefix(state_dict, "", "backbone.backbone.")
#             base_model.load_state_dict(state_dict, strict=False)
#             print(f"Custom weights loaded from {checkpoint_path}")
#
#         # 2. Extract specific stages for the FPN
#         # These keys correspond to the 4 stages of the Swin hierarchy
#         return_nodes = {
#             'features.1': '0',
#             'features.3': '1',
#             'features.5': '2',
#             'features.7': '3',
#         }
#
#         self.body = create_feature_extractor(base_model, return_nodes=return_nodes)
#         if apply_lora:
#             self.body = LoRA.from_module(self.body, rank=lora_rank)
#             freeze_except_lora(self.body)
#             print("LoRA wrapping applied and non-LoRA params frozen")
#             print(f"trainable params: {count_params(self.body)}")
#
#         self.out_channels = [128, 256, 512, 1024]
#
#     def freeze_parameters(self):
#         for param in self.body.parameters():
#             param.requires_grad = False
#
#     def forward(self, x):
#         out = self.body(x)
#         # Permute from [B, H, W, C] to [B, C, H, W]
#         for k in out:
#             out[k] = out[k].permute(0, 3, 1, 2).contiguous()
#         return out

# ALT RESNET BACKBONE
# class AltTorchvisionResNet50Backbone(nn.Module):
#     def __init__(self, checkpoint_path=None):
#         super().__init__()
#
#         self.body = resnet50(weights=None)
#
#         if checkpoint_path:
#             state_dict = torch.load(checkpoint_path, map_location="cpu")
#             state_dict = model_replace_prefix(state_dict, "", "backbone.body.")
#             missing, unexpected = self.body.load_state_dict(state_dict, strict=False)
#             print(f"Custom weights loaded from {checkpoint_path}")
#             print("Missing keys:", missing)
#             print("Unexpected keys:", unexpected)
#
#         # Stem
#         self.conv1 = self.body.conv1
#         self.bn1 = self.body.bn1
#         self.relu = self.body.relu
#         self.maxpool = self.body.maxpool
#
#         # ResNet stages
#         self.layer1 = self.body.layer1  # C2
#         self.layer2 = self.body.layer2  # C3
#         self.layer3 = self.body.layer3  # C4
#         self.layer4 = self.body.layer4  # C5
#
#         # FPN channel sizes for ResNet50
#         self.out_channels = [256, 512, 1024, 2048]
#
#     def freeze_parameters(self):
#         for param in self.body.parameters():
#             param.requires_grad = False
#
#     def forward(self, x):
#         outputs = {}
#
#         # Stem
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         # Stages
#         x = self.layer1(x)
#         outputs["0"] = x   # C2
#
#         x = self.layer2(x)
#         outputs["1"] = x   # C3
#
#         x = self.layer3(x)
#         outputs["2"] = x   # C4
#
#         x = self.layer4(x)
#         outputs["3"] = x   # C5
#
#         return outputs
