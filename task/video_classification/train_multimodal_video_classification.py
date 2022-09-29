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
import itertools
import logging
import random
import os
from functools import partial
from multiprocessing import cpu_count, Pool
from typing import Optional
from sklearn.preprocessing import LabelEncoder

import numpy as np
import torchmetrics
from tqdm import tqdm
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertConfig
from model.modeling_cmt.cross_modal_transformer import CrossTransformerModel
from model.modeling_swin_transformer.video_swin_transformer import SwinTransformer3D
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


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


class MultimodalDataModule(pl.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(self, args, tokenizer, label_encode=None):
        self.args = args
        self.label_encode: LabelEncoder = label_encode
        self.tokenizer = tokenizer
        super().__init__()

    def prepare_data(self) -> None:
        """
        读取视频标注文件，包含训练集文件和测试集文件
        """
        with open(self.args.train_data, "r", encoding="utf-8") as g:
            for line in g:
                ss = line.strip().split()

    def _make_transforms(self, mode: str):
        """
        ##################
        # PTV Transforms #
        ##################
        # Each PyTorchVideo dataset has a "transform" arg. This arg takes a
        # Callable[[Dict], Any], and is used on the output Dict of the dataset to
        # define any application specific processing or augmentation. Transforms can
        # either be implemented by the user application or reused from any library
        # that's domain specific to the modality. E.g. for video we recommend using
        # TorchVision, for audio we recommend TorchAudio.
        #
        # To improve interoperation between domain transform libraries, PyTorchVideo
        # provides a dictionary transform API that provides:
        #   - ApplyTransformToKey(key, transform) - applies a transform to specific modality
        #   - RemoveKey(key) - remove a specific modality from the clip
        #
        # In the case that the recommended libraries don't provide transforms that
        # are common enough for PyTorchVideo use cases, PyTorchVideo will provide them in
        # the same structure as the recommended library. E.g. TorchVision didn't
        # have a RandomShortSideScale video transform so it's been added to PyTorchVideo.
        """

        if self.args.data_type == "video":
            transform = [
                self._video_transform(mode),
                RemoveKey("audio"),
            ]
        elif self.args.data_type == "audio":
            transform = [
                self._audio_transform(),
                RemoveKey("video"),
            ]
        else:
            raise Exception(f"{self.args.data_type} not supported")

        return Compose(transform)

    def _video_transform(self, mode: str):
        """
        This function contains example transforms using both PyTorchVideo and TorchVision
        in the same Callable. For 'train' mode, we use augmentations (prepended with
        'Random'), for 'val' mode we use the respective determinstic function.
        """
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
                label_id=self.label_encode.transform(example.label)
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

    def convert_examples_to_features(self, examples, tokenizer, label_encode, max_length, threads=4):
        """
        多进程的文本处理
        """
        threads = min(threads, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(
                self.convert_single_example,
                tokenizer=tokenizer,
                label_encode=label_encode,
                max_length=max_length
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
    def __init__(self, args):
        self.args = args
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(args.bert_config)
        # 更新bert config
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        # 加载video模型并初始化
        self.video_swin = SwinTransformer3D()
        # 加载文本模型并初始化
        self.bert = BertPreTrainedModel.from_pretrained(args.bert_model,
                                                        config=self.bert_config
                                                        )
        self.co_transformer = CrossTransformerModel(args.max_position_embeddings,
                                                    args.hidden_dropout_prob,
                                                    args.hidden_size,
                                                    args.num_hidden_layers,
                                                    args.num_attention_heads)

    @staticmethod
    def add_model_special_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("VideoClassificationModule")
        parser.add_argument("--max_position_embeddings", type=int, help="cross model ")
        parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="cross model")
        parser.add_argument("--hidden_size", type=int, default=256, help="")
        parser.add_argument("--num_hidden_layers", type=int, default=3, help="")
        parser.add_argument("--num_attention_heads", type=int, default=6, help="")
        return parent_parser

    def forward(self, x, attention_mask, image_mask):  # 定义该模块的forward方法，可以方便推理和预测的接口
        video_feature = self.video_swin(x)
        text_feature = self.bert(x)
        output_feature = torch.concat((video_feature, text_feature), dim=1)

        concat_mask = torch.cat((attention_mask, image_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        image_type_ = torch.ones_like(image_mask)
        concat_type = torch.cat((text_type_, image_type_), dim=1)
        # 构建concat type
        cross_output, pooled_output = self.co_transformer(output_feature, concat_type, concat_mask)
        return cross_output, pooled_output

    def training_step(self, batch, batch_idx):
        """
        batch key and value
           {
               'video': <video_tensor>,
               'token index': <audio_tensor>,
               'attention_mask'
               'label': <action_label>,
           }
        """
        x = batch['video']
        y_hat = self.video_swin(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        # 需要重新定义优化器和学习率
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]


def main():
    """
    To train the ResNet with the Kinetics dataset we construct the two modules above,
    and pass them to the fit function of a pl.Trainer.
    This example can be run either locally (with default parameters) or on a Slurm
    cluster. To run on a Slurm cluster provide the --on_cluster argument.
    """
    setup_logger()
    setup_seed(42)

    pl.trainer.seed_everything()
    parser = argparse.ArgumentParser()
    # Model parameters.
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--arch",
        default="swin_transformer",
        choices=["swin_transformer", "swin_transformer"],
        type=str,
    )

    # Data parameters.
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--video_path_prefix", default="", type=str)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--clip_duration", default=2, type=float)
    parser.add_argument(
        "--data_type", default="video", choices=["video", "audio"], type=str
    )
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
    # 设置默认参数
    parser.set_defaults(
        max_epochs=200,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
        max_position_embeddings=512
    )
    args = parser.parse_args()
    train(args)


def train(args):
    trainer = pl.Trainer.from_argparse_args(args)
    classification_module = VideoClassificationLightningModule(args)
    data_module = MultimodalDataModule(args)
    trainer.fit(classification_module, data_module)


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # tf gpu fix seed, please `pip install tensorflow-determinism` first


if __name__ == "__main__":
    main()
