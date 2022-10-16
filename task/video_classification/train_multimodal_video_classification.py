# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: train_multimodal_video_classification.py
@time: 2022/9/28 10:33
"""

import argparse
import logging
import json
import os
import random
from abc import ABCMeta
from functools import partial
from multiprocessing import cpu_count, Pool
from typing import Optional, Tuple, List, Type, Callable, Any
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorchvideo.data import ClipSampler, make_clip_sampler
from pytorchvideo.data.utils import MultiProcessSampler
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataset import IterableDataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip
)
from tqdm import tqdm
from transformers import BertTokenizer, PreTrainedTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from loguru import logger
from core.snipets import sequence_padding, convert_array_to_tensor
from model.modeling_cmt.cross_modal_transformer import CrossTransformerModel
from model.modeling_swin_transformer.video_swin_transformer import SwinTransformer3D


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


class MultimodalDataset(IterableDataset, metaclass=ABCMeta):
    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
            self,
            input_features: List[Tuple[str, Optional[dict]]],
            clip_sampler: ClipSampler,
            video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = False,
            decoder: str = "pyav",
    ) -> None:
        """
        Args:
            input_features (List[Tuple[str, Optional[dict]]]): List containing multimodal input feature

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, also decode audio from video.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """

        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._input_features = input_features
        self._decoder = decoder

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._input_features, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._input_features)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0
        self.video_path_handler = VideoPathHandler()

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)
                try:
                    input_feature = self._input_features[video_index]
                    info_dict = input_feature.__dict__
                    video = self.video_path_handler.video_from_path(
                        filepath=input_feature.video_name,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    logger.debug(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    continue

            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            if isinstance(clip_start, list):  # multi-clip in each sample

                # Only load the clips once and reuse previously stored clips if there are multiple
                # views for augmentations to perform on the same clips.
                if aug_index[0] == 0:
                    self._loaded_clip = {}
                    loaded_clip_list = []
                    for i in range(len(clip_start)):
                        clip_dict = video.get_clip(clip_start[i], clip_end[i])
                        if clip_dict is None or clip_dict["video"] is None:
                            self._loaded_clip = None
                            break
                        loaded_clip_list.append(clip_dict)

                    if self._loaded_clip is not None:
                        for key in loaded_clip_list[0].keys():
                            self._loaded_clip[key] = [x[key] for x in loaded_clip_list]

            else:  # single clip case

                # Only load the clip once and reuse previously stored clip if there are multiple
                # views for augmentations to perform on the same clip.
                if aug_index == 0:
                    self._loaded_clip = video.get_clip(clip_start, clip_end)

            self._next_clip_start_time = clip_end

            video_is_null = (
                    self._loaded_clip is None or self._loaded_clip["video"] is None
            )
            if (
                    is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
            ) or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._next_clip_start_time = 0.0
                self._clip_sampler.reset()
                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue

            frames = self._loaded_clip["video"]
            audio_samples = self._loaded_clip["audio"]
            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

                # User can force dataset to continue by returning None in transform.
                if sample_dict is None:
                    continue

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


class MultimodalDataModule(pl.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(self, args, tokenizer: Optional[PreTrainedTokenizer], label_encode):
        self.args = args
        self.tokenizer = tokenizer
        self.label_encode = label_encode
        super().__init__()

    def setup(self, stage: Optional[str] = None) -> None:
        """
          读取视频标注文件，包含训练集文件和测试集文件
        """
        if stage == "fit":
            train_example, test_example = self._load_raw_data(val=True)
            # print(train_example[0].label)
            train_feature = self.convert_examples_to_features(train_example,
                                                              self.args.max_length,
                                                              mode="train")
            test_feature = self.convert_examples_to_features(test_example,
                                                             self.args.max_length,
                                                             mode="train")
            self.train_dataset = MultimodalDataset(input_features=train_feature,
                                                   clip_sampler=make_clip_sampler("random",
                                                                                  self.args.clip_duration),
                                                   transform=self._make_transforms(mode="train"),
                                                   video_sampler=torch.utils.data.sampler.RandomSampler)
            self.val_dataset = MultimodalDataset(input_features=test_feature,
                                                 clip_sampler=make_clip_sampler("random",
                                                                                self.args.clip_duration),
                                                 transform=self._make_transforms(mode="train"),
                                                 video_sampler=torch.utils.data.sampler.RandomSampler)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        for k, v in batch.items():
            batch[k] = v.to(device)

        return batch

    def train_dataloader(self):

        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.args.batch_size,
                                           collate_fn=self._collate_fn,
                                           num_workers=self.args.num_workers
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.args.batch_size,
                                           collate_fn=self._collate_fn,
                                           num_workers=self.args.num_workers

                                           )

    def _load_raw_data(self, val=False):
        train_example = []
        with open(self.args.train_data, "r", encoding="utf-8") as g:
            train_data = json.load(g)
            for index, data in enumerate(train_data):
                train_example.append(InputExample(
                    guid=f"train-{index}",
                    video_name=os.path.join(self.args.video_path_prefix, data["video"]),
                    text=data["text"],
                    # label 为二维数组
                    label=data["label"]
                ))

        if val:
            test_example = []
            with open(self.args.test_data, "r", encoding="utf-8") as g:
                test_data = json.load(g)
                for index, data in enumerate(test_data):
                    test_example.append(InputExample(
                        guid=f"test-{index}",
                        video_name=os.path.join(self.args.video_path_prefix, data["video"]),
                        text=data["text"],
                        label=data["label"]
                    ))
            return train_example, test_example
        return train_example

    def _collate_fn(self, data):
        batch_token_ids, batch_segment_ids, batch_input_mask, batch_labels, guids, videos = [], [], [], [], [], []
        for feature in data:
            guids.append(feature["guid"])
            videos.append(feature["video"])
            batch_token_ids.append(feature["text_input_ids"])
            batch_input_mask.append(feature["text_input_mask"])
            batch_segment_ids.append(feature["text_segment_ids"])
            batch_labels.append(np.array(feature["label_id"]))

        batch_token_ids = sequence_padding(batch_token_ids, value=self.tokenizer.pad_token_id)
        batch_input_mask = sequence_padding(batch_input_mask)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        # 先转化为array再转为tensor
        return {
            "videos": torch.stack(videos, dim=0),
            "token_input_ids": convert_array_to_tensor(batch_token_ids),
            "token_type_ids": convert_array_to_tensor(batch_segment_ids),
            "token_attention_mask": convert_array_to_tensor(batch_input_mask),
            "labels": convert_array_to_tensor(np.array(batch_labels))
        }

    def _make_transforms(self, mode: str):
        transform = [
            self._video_transform(mode),
            RemoveKey("audio"),
        ]
        return Compose(transform)

    def _video_transform(self, mode: str):
        args = self.args
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(args.video_num_subsampled),
                    Normalize(args.video_means, args.video_stds),
                ]
                + (
                    [
                        RandomShortSideScale(
                            min_size=args.video_min_short_side_scale,
                            max_size=args.video_max_short_side_scale,
                        ),
                        RandomCrop(args.video_crop_size),
                        RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                    ]
                    if mode == "train"
                    else [
                        ShortSideScale(args.video_min_short_side_scale),
                        CenterCrop(args.video_crop_size),
                    ]
                )
            ),
        )

    def convert_single_example(self, example: InputExample,
                               max_length: int = 128,
                               mode: str = "train"):

        sentence = "[SEP]".join([example.text])
        encoder = self.tokenizer(sentence, max_length=max_length, truncation=True)
        input_ids = encoder.input_ids
        segment_ids = encoder.token_type_ids
        input_mask = encoder.attention_mask
        if mode == "train":
            assert self.label_encode is not None
            feature = InputFeature(
                guid=example.guid,
                text_input_ids=input_ids,
                text_input_mask=input_mask,
                text_segment_ids=segment_ids,
                video_name=example.video_name,
                # label transform添加一个维度
                label_id=self.label_encode.transform([example.label])
            )
        else:
            feature = InputFeature(
                guid=example.guid,
                text_input_ids=input_ids,
                text_input_mask=input_mask,
                text_segment_ids=segment_ids,
                video_name=example.video_name,
            )
        return feature

    def convert_examples_to_features(self, examples, max_length, threads=4, mode="train"):
        """
        多进程的文本处理
        """
        threads = min(threads, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(
                self.convert_single_example,
                max_length=max_length,
                mode=mode
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=32),
                    total=len(examples),
                    desc="convert examples to features",
                )
            )
        return features


class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(args.bert_model)
        # 更新bert config
        # TODO
        self.acc = torchmetrics.Accuracy(num_classes=2, top_k=1)
        # 加载video模型并初始化
        self.video_swin = SwinTransformer3D(pretrained=args.pretrained_video_model)
        # 加载文本模型并初始化
        self.bert = BertModel.from_pretrained(args.bert_model,
                                              config=self.bert_config
                                              )
        self.co_transformer = CrossTransformerModel(args.max_position_embeddings,
                                                    args.hidden_dropout_prob,
                                                    args.hidden_size,
                                                    args.num_hidden_layers,
                                                    args.num_attention_heads)
        # 视频特征处理模块
        if self.args.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.args.dropout_ratio)
        else:
            self.dropout = None
        if self.args.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        else:
            self.avg_pool = None
        self.linear = nn.Linear(in_features=args.hidden_size, out_features=2)
        # 加载预训练的权重
        self.init_weights()
        self.save_hyperparameters()

    def init_weights(self):
        logger.info("loading video pretrain model")
        self.video_swin.init_weights()

    @staticmethod
    def add_model_special_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("VideoClassificationModule")
        parser.add_argument("--dropout_ratio", type=float, help="swin transformer head drop out ratio")
        parser.add_argument("--spatial_type", type=str, default="avg", help=" Pooling type in spatial dimension")
        parser.add_argument("--max_position_embeddings", type=int, help="cross model ")
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="cross model")
        parser.add_argument("--hidden_size", type=int, default=768, help="")
        parser.add_argument("--num_hidden_layers", type=int, default=12, help="")
        parser.add_argument("--num_attention_heads", type=int, default=12, help="")
        return parent_parser

    def forward(self,
                video_inputs,
                text_inputs_ids,
                text_attention_mask,
                text_token_type_ids):  # 定义该模块的forward方法，可以方便推理和预测的接口
        video_feature = self.video_swin(video_inputs)
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            video_feature = self.avg_pool(video_feature)
        # [N, in_channels, 1, 7, 7]
        if self.dropout is not None:
            video_feature = self.dropout(video_feature)
        # [N, in_channels, 49]
        video_feature = video_feature.view(video_feature.shape[0], -1, 7 * 7)
        video_feature = torch.transpose(video_feature, 1, 2)
        # 视频特征处理
        # video_feature shape=[N , 49, D=768 ]
        text_feature = self.bert(text_inputs_ids, text_attention_mask, text_token_type_ids).last_hidden_state
        # text_feature shape= [N,sequence length ,768]
        # concat text token and image token
        output_feature = torch.cat((text_feature, video_feature), dim=1)
        # 构建image序列mask
        # image_attention_mask = torch.ones(video_feature.size(0), video_feature.size(1)).type_as(text_inputs_ids)
        #
        # concat_mask = torch.cat((text_attention_mask, image_attention_mask), dim=1)
        # text_type_ = torch.zeros_like(text_attention_mask)
        # image_type_ = torch.ones_like(image_attention_mask)
        # concat_type = torch.cat((text_type_, image_type_), dim=1)
        # # 构建concat type
        cross_output, pooled_output = self.co_transformer(output_feature)
        return cross_output, pooled_output

    def training_step(self, batch, batch_idx):
        labels = batch["labels"].squeeze().to(dtype=torch.long)
        _, y = self.forward(video_inputs=batch["videos"],
                            text_inputs_ids=batch["token_input_ids"],
                            text_attention_mask=batch["token_attention_mask"],
                            text_token_type_ids=batch["token_type_ids"])
        y_hat = self.linear(y)
        loss = F.cross_entropy(y_hat, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        label = batch["labels"].squeeze().to(dtype=torch.long)
        _, y = self.forward(video_inputs=batch["videos"],
                            text_inputs_ids=batch["token_input_ids"],
                            text_attention_mask=batch["token_attention_mask"],
                            text_token_type_ids=batch["token_type_ids"])
        y_hat = self.linear(y)
        loss = F.cross_entropy(y_hat, label)
        output = F.softmax(y_hat, dim=-1)
        self.log("val_loss", loss, prog_bar=True)
        return output, label

    def validation_epoch_end(self, outputs) -> None:
        # print(outputs)
        preds = []
        targets = []
        for pred, target in outputs:
            preds.extend(torch.split(pred, 1, dim=0))
            targets.extend(torch.split(target, 1, dim=0))
        p = torch.stack(preds, dim=0)
        p = torch.reshape(p, shape=[-1, p.shape[-1]])
        t = torch.stack(targets, dim=0)
        t = torch.reshape(t, shape=[-1])
        acc = self.acc(p, t)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return

    def configure_optimizers(self):
        # 需要重新定义优化器和学习率

        optimizer = torch.optim.SGD(
            [
                {"params": self.video_swin.parameters(), "lr": self.args.learning_rate_video},
                {"params": self.bert.parameters(), "lr": self.args.learning_rate_bert},
                {"params": self.linear.parameters()},
                {"params": self.co_transformer.parameters()}
            ],
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]


def setup_seed(seed):
    pl.seed_everything(seed)
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    """
    main function
    """

    setup_seed(42)
    parser = argparse.ArgumentParser()
    # Model parameters.
    parser.add_argument("--bert_model", type=str, help="text bert model path")
    parser.add_argument("--learning-rate-bert", type=float, help="bert model learning rate")
    parser.add_argument("--pretrained_video_model", type=str, help="pretrained video model")
    parser.add_argument("--learn-rate-video", type=float, help="video swin transformer pretrained learning rate")
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, help="default learning rate")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    # Data parameters.
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)

    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--video_path_prefix", type=str, help="video data directory")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--clip_duration", default=2, type=float)
    parser.add_argument("--video_num_subsampled", default=8, type=int)
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--video_crop_size", default=224, type=int)
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
    parser.add_argument("--video_horizontal_flip_p", default=0.5, type=float)

    # Trainer parameters.
    parser = pl.Trainer.add_argparse_args(parser)

    # 添加模型参数
    parser = VideoClassificationLightningModule.add_model_special_args(parser)

    # 设置部分默认参数
    parser.set_defaults(
        max_epochs=10,
        replace_sampler_ddp=False,
        max_position_embeddings=512,
        dropout_ratio=0.5,
        learning_rate_bert=1e-5,
        learning_rate_video=1e-3
    )
    args = parser.parse_args()
    checkpoint_callback = ModelCheckpoint(dirpath=args.default_root_dir,
                                          monitor='train_loss',
                                          filename='multimodal_video-{epoch:02d}-{train_loss:.2f}')
    early_stop = EarlyStopping("val_acc", mode="max", patience=5, min_delta=0.01, verbose=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    label_encode = LabelEncoder()
    label_encode.fit(['0', '1'])

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[LearningRateMonitor(), checkpoint_callback, early_stop])
    classification_module = VideoClassificationLightningModule(args)
    data_module = MultimodalDataModule(args=args, tokenizer=tokenizer, label_encode=label_encode)
    trainer.fit(classification_module, data_module, )


if __name__ == "__main__":
    main()
