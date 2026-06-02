##
import os
import yaml
from utils import event_to_yml, yaml_to_dict_parser, write_dict_to_yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


path = "/home/jazib/projects/data/oam-tcd-coco-style-1024/"
file_list = [
    'coco_annotations_fold_0.json',
    'coco_annotations_fold_1.json',
    'coco_annotations_fold_2.json',
    'coco_annotations_fold_3.json',
    'coco_annotations_fold_4.json',
    'coco_annotations_test.json'
]


def create_fold_dicts(path, file_list):
    cv_dict = {}
    for file in file_list:
        fold_dict = {}
        with open(os.path.join(path, file), 'r') as fr:
            data = json.load(fr)
        data = data['images']
        for img_d in data:
            if str(img_d['biome_id']) not in fold_dict.keys():
                fold_dict[str(img_d['biome_id'])] = 1
            else:
                fold_dict[str(img_d['biome_id'])] += 1
        cv_dict[file] = fold_dict
    return cv_dict


cross_val_dict = create_fold_dicts(path, file_list)
##
sum_cross_val_dict = {}
for k1, v1 in cross_val_dict.items():
    sum_cross_val_dict[k1] = sum(v1.values())
##
df_dict = {}
for f, d in cross_val_dict.items():
    df_dict[f]=(pd.DataFrame([d]))
##
new_cv_df = pd.concat(list(df_dict.values())).set_axis(list(df_dict.keys()))

new_cv_df = new_cv_df[['-1', '1', '2', '3', '4', '5', '6', '7', '8', '98', '9', '10', '11', '12', '13', '14']]

new_cv_df = new_cv_df.fillna(0)

new_cv_df = new_cv_df.astype(int)

new_cv_df['8'] = new_cv_df['8'] + new_cv_df['98']
new_cv_df = new_cv_df.drop(columns=['98'])

new_cv_df = new_cv_df.transpose()
##
new_cv_df['row_sum'] = new_cv_df.sum(axis=1)
new_cv_df.loc['col_sum'] = new_cv_df.sum(axis=0)
##
new_cv_df.to_latex("./statistics/table_dist.tex", index=True)