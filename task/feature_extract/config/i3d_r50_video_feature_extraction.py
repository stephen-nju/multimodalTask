# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: i3d_r50_video_feature_extraction.py
@time: 2022/11/4 15:11
"""

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/resnet50-0676ba61.pth',
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(feature_extraction=True))

dataset_type = 'VideoDataset'
img_norm_cfg = dict(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), to_bgr=False)

# pipline = Compose([
#     DecordInit(),
#     SampleFrames(clip_len=self.args.clip_len, frame_interval=10, num_clips=self.args.num_clips),
#     DecordDecode(),
#     Resize(scale=(-1, 256)),
#     RandomResizedCrop(),
#     Resize(scale=(224, 224), keep_ratio=False),
#     Flip(flip_ratio=0.5),
#     Normalize(mean=self.args.image_means, std=self.args.image_stds, to_bgr=False),
#     FormatShape(input_format='NCTHW'),
#     ToTensor(['imgs']),


test_pipeline = [
    dict(type='DecordInit', num_threads=1),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))
