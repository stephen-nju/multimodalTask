# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: c3d_sports1m_video_feature_extraction.py
@time: 2022/8/29 15:58
"""
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(feature_extraction=True))

#  raw frame dataset setting
# dataset_type = 'RawframeDataset'
# img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)
# test_pipeline = [
#     dict(
#         type='SampleFrames',
#         clip_len=16,
#         frame_interval=1,
#         num_clips=10,
#         test_mode=True),
#     dict(type='RawFrameDecode'),
#     dict(type='Resize', scale=(128, 171)),
#     dict(type='CenterCrop', crop_size=112),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs', 'label'])
# ]

dataset_type = 'VideoDataset'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)

test_pipeline = [
    dict(type='OpenCVInit', num_threads=1),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
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
