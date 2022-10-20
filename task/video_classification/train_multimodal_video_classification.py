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
import json
import os
import random
from functools import partial
from multiprocessing import cpu_count, Pool
from typing import Optional, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from loguru import logger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.nn.init import trunc_normal_
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from transformers import BertTokenizer, PreTrainedTokenizer
from transformers.models.bert.modeling_bert import BertConfig, BertModel

from core.datasets.data_loading import DecordInit, SampleFrames, DecordDecode, ToTensor, FormatShape, Compose
from core.datasets.multimodal_dataset import MultimodalDataset, InputExample, InputFeature
from core.datasets.transform import RandomResizedCrop, CenterCrop
from core.datasets.transform import Resize, Flip, Normalize
from core.snipets import sequence_padding, convert_array_to_tensor
from model.modeling_cmt.cross_modal_transformer import CrossTransformerModel
from model.modeling_swin_transformer.video_swin_transformer import SwinTransformer3D


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
                                                   pipeline=self._make_train_pipeline(),
                                                   )
            self.val_dataset = MultimodalDataset(input_features=test_feature,
                                                 pipeline=self._make_val_pipeline(),
                                                 )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        for k, v in batch.items():
            batch[k] = v.to(device)

        return batch

    def train_dataloader(self):

        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.args.batch_size,
                                           collate_fn=self._collate_fn,
                                           num_workers=self.args.num_workers,
                                           shuffle=True
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
            videos.append(torch.squeeze(feature["imgs"], dim=0))
            batch_token_ids.append(feature["text_input_ids"])
            batch_input_mask.append(feature["text_input_mask"])
            batch_segment_ids.append(feature["text_segment_ids"])
            batch_labels.append(np.array(feature["label"]))

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

    def _make_train_pipeline(self):
        """
        video加载和数据预处理
        """
        pipline = Compose([
            DecordInit(),
            SampleFrames(clip_len=self.args.clip_len, frame_interval=10, num_clips=self.args.num_clips),
            DecordDecode(),
            Resize(scale=(-1, 256)),
            RandomResizedCrop(),
            Resize(scale=(224, 224), keep_ratio=False),
            Flip(flip_ratio=0.5),
            Normalize(mean=self.args.image_means, std=self.args.image_stds, to_bgr=False),
            FormatShape(input_format='NCTHW'),
            ToTensor(['imgs']),

        ])
        logger.info(pipline)
        return pipline

    def _make_val_pipeline(self):
        pipline = Compose([
            DecordInit(),
            SampleFrames(clip_len=self.args.clip_len, frame_interval=10, num_clips=self.args.num_clips),
            DecordDecode(),
            Resize(scale=(-1, 256)),
            CenterCrop(crop_size=224),
            Flip(flip_ratio=0.5),
            Normalize(mean=self.args.image_means, std=self.args.image_stds, to_bgr=False),
            FormatShape(input_format='NCTHW'),
            ToTensor(['imgs']),

        ])
        return pipline

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
        self.train_acc = torchmetrics.Accuracy(num_classes=2, top_k=1)
        # 加载video模型并初始化
        self.video_swin = SwinTransformer3D(patch_size=args.patch_size,
                                            patch_norm=True,
                                            drop_path_rate=args.drop_path_rate,
                                            window_size=(8, 7, 7),
                                            pretrained=args.pretrained_video_model)
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
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(in_features=args.hidden_size, out_features=2)
        # 加载预训练的权重
        self.init_weights()
        self.save_hyperparameters(args)

    def init_weights(self):
        torch.nn.init.trunc_normal_(self.linear.weight, std=0.02)
        # co transformer 初始化
        self.co_transformer.initialize_parameters()
        logger.info("loading video pretrain model")
        self.video_swin.init_weights()

    @staticmethod
    def add_model_special_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("VideoClassificationModule")
        parser.add_argument("--patch_size", type=tuple)
        parser.add_argument("--drop_path_rate", type=float)
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

    def on_train_start(self) -> None:
        # 训练开始时候 记录模型
        self.logger.log_graph(self)

    def training_step(self, batch, batch_idx):
        labels = batch["labels"].squeeze().to(dtype=torch.long)
        _, y = self.forward(video_inputs=batch["videos"],
                            text_inputs_ids=batch["token_input_ids"],
                            text_attention_mask=batch["token_attention_mask"],
                            text_token_type_ids=batch["token_type_ids"])
        y = self.dropout(y)
        y_hat = self.linear(y)
        loss = F.cross_entropy(y_hat, labels)
        self.train_acc.update(y_hat, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("learning rate", self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])

        return loss

    def on_train_epoch_end(self) -> None:
        tensorboard = self.logger.experiment
        # 查看线性映射层的gradient
        for name, param in self.linear.named_parameters():
            tensorboard.add_histogram(name + "_grad", param.grad)

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"].squeeze().to(dtype=torch.long)
        _, y = self.forward(video_inputs=batch["videos"],
                            text_inputs_ids=batch["token_input_ids"],
                            text_attention_mask=batch["token_attention_mask"],
                            text_token_type_ids=batch["token_type_ids"])
        y = self.dropout(y)
        y_hat = self.linear(y)  # y_hat=[batch_size,2]
        loss = F.cross_entropy(y_hat, labels)
        output = F.softmax(y_hat, dim=-1)  # output=[batch_size,num_classes=2]
        self.log("val_loss", loss, prog_bar=True)

        return output, labels

    def validation_epoch_end(self, outputs) -> None:
        preds = []
        targets = []
        # pred=[batch_size,num_classes=2]
        # target=[batch_size]
        for pred, target in outputs:
            preds.extend(torch.split(pred, 1, dim=0))
            targets.extend(torch.split(target, 1, dim=0))
        p = torch.stack(preds, dim=0)
        p = torch.squeeze(p, dim=1)
        t = torch.stack(targets, dim=0)
        t = torch.squeeze(t, dim=1)
        acc = self.acc(p, t)
        p = np.argmax(p.cpu().numpy(), axis=1)
        print(classification_report(t.cpu().numpy(), p, digits=4))
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return

    def configure_optimizers(self):
        # 需要重新定义优化器和学习率
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(
                [
                    {"params": self.video_swin.parameters(), "lr": self.args.learning_rate_video,
                     "weight_decay": self.args.weight_decay},
                    {"params": self.bert.parameters(), "lr": self.args.learning_rate_bert},
                    {"params": self.linear.parameters(), "weight_decay": self.args.weight_decay * 100},
                    {"params": self.co_transformer.parameters()}
                ],
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                [
                    {"params": self.video_swin.parameters(), "lr": self.args.learning_rate_video,
                     "weight_decay": self.args.weight_decay},
                    {"params": self.bert.parameters(), "lr": self.args.learning_rate_bert},
                    {"params": self.linear.parameters(), "weight_decay": self.args.weight_decay * 100},
                    {"params": self.co_transformer.parameters()}
                ],
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )


        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]


def setup_seed(seed):
    pl.seed_everything(seed)
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--optimizer_name", type=str, choices=["Adam", "SGD"], help="model optimizer")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=2e-4, type=float)
    # Data parameters.
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)

    # video data processing
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--video_path_prefix", type=str, help="video data directory")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", type=int, help="train batch size")
    parser.add_argument("--clip_len", type=int, help="video clip length")
    parser.add_argument("--num_clips", type=int, help="num clip to sample")
    parser.add_argument("--image_means", type=tuple)
    parser.add_argument("--image_stds", type=tuple)

    # Trainer parameters.
    parser = pl.Trainer.add_argparse_args(parser)

    # 添加模型参数
    parser = VideoClassificationLightningModule.add_model_special_args(parser)

    # 设置部分默认参数
    parser.set_defaults(
        optimizer_name="SGD",
        max_epochs=10,
        replace_sampler_ddp=False,
        max_position_embeddings=256,
        # =========model default param======
        patch_size=(4, 4, 4),
        drop_path_rate=0.2,
        dropout_ratio=0.3,
        learning_rate_bert=1e-5,
        learning_rate_video=1e-3,
        # =======video data process=============
        clip_len=32,
        num_clips=1,
        image_means=(123.675, 116.28, 103.53),
        image_stds=(58.395, 57.12, 57.375),

    )
    args = parser.parse_args()
    checkpoint_callback = ModelCheckpoint(dirpath=args.default_root_dir,
                                          monitor='train_loss',
                                          filename='multimodal_video-{epoch:02d}-{train_loss:.2f}')
    early_stop = EarlyStopping("val_acc", mode="max", patience=5, min_delta=0.01, verbose=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    label_encode = LabelEncoder()
    label_encode = label_encode.fit(['0', '1'])

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[LearningRateMonitor(logging_interval="step"),
                                                             checkpoint_callback, early_stop])
    classification_module = VideoClassificationLightningModule(args)

    data_module = MultimodalDataModule(args=args, tokenizer=tokenizer, label_encode=label_encode)
    trainer.fit(classification_module, data_module)
    # data_module.setup("fit")
    #
    # for d in data_module.train_dataset:
    #     print(d["imgs"])


if __name__ == "__main__":
    main()
