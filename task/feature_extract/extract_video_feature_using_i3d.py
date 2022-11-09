# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: extract_video_feature_using_i3d.py
@time: 2022/11/2 19:51
"""
import shutil

import torch
import os
import numpy as np
from mmaction.apis import init_recognizer, inference_recognizer
import re
from tqdm import tqdm

FILE_PATTERN = re.compile(r"\d+\.mp4")

config_file = './task/feature_extract/config/i3d_r50_video_feature_extraction.py'
# 从模型库中下载检测点，并把它放到 `checkpoints/` 文件夹下
checkpoint_file = '/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth'

# 指定设备
device = 'gpu'  # or 'cpu'
device = torch.device(device)
# 根据配置文件和检查点来建立模型
model = init_recognizer(config_file, checkpoint_file, device=device)

video_root = "/home/nlpbigdata/local_disk/video"
video_feature_root = "/home/nlpbigdata/local_disk/video_feature"
for video in tqdm(os.listdir(video_root)):
    video_dir = os.path.join(video_root, video)
    if os.path.isdir(video_dir):
        feature_dir = os.path.join(video_feature_root, video)
        if not os.path.exists(feature_dir):
            os.mkdir(feature_dir)
        else:
            shutil.rmtree(feature_dir)
            os.mkdir(feature_dir)
        for video_span in os.listdir(video_dir):
            if re.match(FILE_PATTERN, video_span):
                video_span_path = os.path.join(video_dir, video_span)
                if os.path.isfile(video_span_path):
                    results, returned_features = inference_recognizer(model, video_span_path, outputs="backbone")
                    feat = returned_features["backbone"].detach().cpu().numpy()
                    np.save(os.path.join(feature_dir, f"{video_span}.npy"), feat)
