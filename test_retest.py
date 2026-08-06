##
import torch
import os


path = "/data_hdd/jazibmodels/Fine_Tuning_Strategies/s1_convnext_lora_full/last.ckpt"
ckpt = torch.load(path, map_location="cpu")

##
pth_od = "./lora_keys.txt"
od_keys = list(ckpt['state_dict'].keys())

with open(pth_od, 'w') as f2:
    for i in od_keys:
        f2.write("%s\n" % i)