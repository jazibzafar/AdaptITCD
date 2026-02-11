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