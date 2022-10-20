# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: multimodal_dataset.py
@time: 2022/10/17 19:07
"""
import argparse
import copy
import functools
from abc import ABCMeta
from typing import Optional, Tuple, List, Type, Callable, Any

import torch
from pytorchvideo.data import ClipSampler
from pytorchvideo.data.utils import MultiProcessSampler
from pytorchvideo.data.video import VideoPathHandler
from torch.utils.data import IterableDataset
from loguru import logger
from torch.utils.data.dataset import T_co


class InputExample(object):
    """
    原始输入的基本格式
    """

    def __init__(self, guid, text, video_name, label=None):
        self.guid = guid
        self.text = text
        self.video_name = video_name
        self.label = label


class InputFeature(object):
    """
    特征输入的基本形式
    """

    def __init__(self,
                 guid,
                 text_input_ids,
                 text_input_mask,
                 text_segment_ids,
                 video_name,
                 label_id=None):
        # label id 默认为None，区分训练和预测
        self.guid = guid
        self.text_input_ids = text_input_ids
        self.text_input_mask = text_input_mask
        self.text_segment_ids = text_segment_ids
        self.video_name = video_name
        # 视频特征运行时处理，保留视频地址
        self.label_id = label_id


class MultimodalDataset(torch.utils.data.Dataset):
    """
           start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1
    """

    def __init__(
            self,
            input_features: List[Tuple[str, Optional[dict]]],
            pipeline: Optional[Callable] = None,
            modality='RGB',
            start_index: int = 0,
    ) -> None:
        self._pipeline = pipeline
        self._input_features = input_features
        self.start_index = start_index
        self.modality = modality

    def __getitem__(self, index):
        input_feature = self._input_features[index]
        feature_dict = copy.deepcopy(input_feature.__dict__)
        feature_dict["filename"] = feature_dict["video_name"]
        feature_dict["label"] = feature_dict["label_id"]
        feature_dict["start_index"] = self.start_index
        feature_dict["modality"] = self.modality

        feature_dict = self._pipeline(feature_dict)
        return feature_dict

    def __len__(self):
        return len(self._input_features)

#
# class MultimodalDataset(IterableDataset, metaclass=ABCMeta):
#     _MAX_CONSECUTIVE_FAILURES = 10
#
#     def __init__(
#             self,
#             input_features: List[Tuple[str, Optional[dict]]],
#             clip_sampler: ClipSampler,
#             video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
#             transform: Optional[Callable[[dict], Any]] = None,
#             decode_audio: bool = False,
#             decoder: str = "pyav",
#     ) -> None:
#         """
#         Args:
#             input_features (List[Tuple[str, Optional[dict]]]): List containing multimodal input feature
#
#             clip_sampler (ClipSampler): Defines how clips should be sampled from each
#                 video. See the clip sampling documentation for more information.
#
#             video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
#                 video container. This defines the order videos are decoded and,
#                 if necessary, the distributed split.
#
#             transform (Callable): This callable is evaluated on the clip output before
#                 the clip is returned. It can be used for user defined preprocessing and
#                 augmentations on the clips. The clip output format is described in __next__().
#
#             decode_audio (bool): If True, also decode audio from video.
#
#             decoder (str): Defines what type of decoder used to decode a video. Not used for
#                 frame videos.
#         """
#
#         self._decode_audio = decode_audio
#         self._transform = transform
#         self._clip_sampler = clip_sampler
#         self._input_features = input_features
#         self._decoder = decoder
#
#         # If a RandomSampler is used we need to pass in a custom random generator that
#         # ensures all PyTorch multiprocess workers have the same random seed.
#         self._video_random_generator = None
#         if video_sampler == torch.utils.data.RandomSampler:
#             self._video_random_generator = torch.Generator()
#             self._video_sampler = video_sampler(
#                 self._input_features, generator=self._video_random_generator
#             )
#         else:
#             self._video_sampler = video_sampler(self._input_features)
#
#         self._video_sampler_iter = None  # Initialized on first call to self.__next__()
#
#         # Depending on the clip sampler type, we may want to sample multiple clips
#         # from one video. In that case, we keep the store video, label and previous sampled
#         # clip time in these variables.
#         self._loaded_video_label = None
#         self._loaded_clip = None
#         self._next_clip_start_time = 0.0
#         self.video_path_handler = VideoPathHandler()
#
#     @property
#     def video_sampler(self):
#         """
#         Returns:
#             The video sampler that defines video sample order. Note that you'll need to
#             use this property to set the epoch for a torch.utils.data.DistributedSampler.
#         """
#         return self._video_sampler
#
#     @property
#     def num_videos(self):
#         """
#         Returns:
#             Number of videos in dataset.
#         """
#         return len(self.video_sampler)
#
#     def __next__(self) -> dict:
#         """
#         Retrieves the next clip based on the clip sampling strategy and video sampler.
#
#         Returns:
#             A dictionary with the following format.
#
#             .. code-block:: text
#
#                 {
#                     'video': <video_tensor>,
#                     'label': <index_label>,
#                     'video_label': <index_label>
#                     'video_index': <video_index>,
#                     'clip_index': <clip_index>,
#                     'aug_index': <aug_index>,
#                 }
#         """
#         if not self._video_sampler_iter:
#             # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
#             self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))
#
#         for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
#             # Reuse previously stored video if there are still clips to be sampled from
#             # the last loaded video.
#             if self._loaded_video_label:
#                 video, info_dict, video_index = self._loaded_video_label
#             else:
#                 video_index = next(self._video_sampler_iter)
#                 try:
#                     input_feature = self._input_features[video_index]
#                     info_dict = input_feature.__dict__
#                     video = self.video_path_handler.video_from_path(
#                         filepath=input_feature.video_name,
#                         decode_audio=self._decode_audio,
#                         decoder=self._decoder,
#                     )
#                     self._loaded_video_label = (video, info_dict, video_index)
#                 except Exception as e:
#                     logger.debug(
#                         "Failed to load video with error: {}; trial {}".format(
#                             e,
#                             i_try,
#                         )
#                     )
#                     continue
#
#             (
#                 clip_start,
#                 clip_end,
#                 clip_index,
#                 aug_index,
#                 is_last_clip,
#             ) = self._clip_sampler(
#                 self._next_clip_start_time, video.duration, info_dict
#             )
#
#             if isinstance(clip_start, list):  # multi-clip in each sample
#
#                 # Only load the clips once and reuse previously stored clips if there are multiple
#                 # views for augmentations to perform on the same clips.
#                 if aug_index[0] == 0:
#                     self._loaded_clip = {}
#                     loaded_clip_list = []
#                     for i in range(len(clip_start)):
#                         clip_dict = video.get_clip(clip_start[i], clip_end[i])
#                         if clip_dict is None or clip_dict["video"] is None:
#                             self._loaded_clip = None
#                             break
#                         loaded_clip_list.append(clip_dict)
#
#                     if self._loaded_clip is not None:
#                         for key in loaded_clip_list[0].keys():
#                             self._loaded_clip[key] = [x[key] for x in loaded_clip_list]
#
#             else:  # single clip case
#
#                 # Only load the clip once and reuse previously stored clip if there are multiple
#                 # views for augmentations to perform on the same clip.
#                 if aug_index == 0:
#                     self._loaded_clip = video.get_clip(clip_start, clip_end)
#
#             self._next_clip_start_time = clip_end
#
#             video_is_null = (
#                     self._loaded_clip is None or self._loaded_clip["video"] is None
#             )
#             if (
#                     is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
#             ) or video_is_null:
#                 # Close the loaded encoded video and reset the last sampled clip time ready
#                 # to sample a new video on the next iteration.
#                 self._loaded_video_label[0].close()
#                 self._loaded_video_label = None
#                 self._next_clip_start_time = 0.0
#                 self._clip_sampler.reset()
#                 if video_is_null:
#                     logger.debug(
#                         "Failed to load clip {}; trial {}".format(video.name, i_try)
#                     )
#                     continue
#
#             frames = self._loaded_clip["video"]
#             audio_samples = self._loaded_clip["audio"]
#             sample_dict = {
#                 "video": frames,
#                 "video_name": video.name,
#                 "video_index": video_index,
#                 "clip_index": clip_index,
#                 "aug_index": aug_index,
#                 **info_dict,
#                 **({"audio": audio_samples} if audio_samples is not None else {}),
#             }
#             if self._transform is not None:
#                 sample_dict = self._transform(sample_dict)
#
#                 # User can force dataset to continue by returning None in transform.
#                 if sample_dict is None:
#                     continue
#
#             return sample_dict
#         else:
#             raise RuntimeError(
#                 f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
#             )
#
#     def __iter__(self):
#         self._video_sampler_iter = None  # Reset video sampler
#
#         # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
#         # same seed for each worker's RandomSampler generator. The workers at each
#         # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
#         # which we can use for this seed.
#         worker_info = torch.utils.data.get_worker_info()
#         if self._video_random_generator is not None and worker_info is not None:
#             base_seed = worker_info.seed - worker_info.id
#             self._video_random_generator.manual_seed(base_seed)
#
#         return self
