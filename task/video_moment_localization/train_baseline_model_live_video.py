# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: train_baseline_model_live_video.py
@time: 2022/11/1 9:20
"""
import argparse
import dataclasses
import json
import random
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count, Pool
from typing import Optional, Any, List, Tuple
import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, BertModel

from core.loss.focal_loss import FocalLoss
from core.metrics.video_moment_local_metric.video_moment_loc_iou import mIOU
from core.snipets import sequence_padding, convert_array_to_tensor
from model.modeling_vml.video_moment_loc_base_model import SimpleVideoMomentLoc


@dataclass
class InputExample:
    qas_id: Any
    query_text: str
    length: int
    video_name: str
    video_feature_path: str
    text_feature_path: str
    start_position: List = dataclasses.field(default=None)
    end_position: List = dataclasses.field(default=None)


@dataclass
class InputFeature:
    qas_id: Any
    query_token_ids: List
    length: int
    video_name: str
    video_feature_path: str
    text_feature_path: str
    start_position: List = dataclasses.field(default=None)
    end_position: List = dataclasses.field(default=None)


class FeatureDataset(Dataset):

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature: InputFeature = self.features[index]
        qas_id = feature.qas_id
        seq_length = feature.length
        query_token_ids = feature.query_token_ids
        video_feature = np.load(feature.video_feature_path)
        text_feature = np.load(feature.text_feature_path)
        start_position = feature.start_position
        end_position = feature.end_position
        start_label = np.zeros([seq_length], dtype=np.int)
        end_label = np.zeros([seq_length], dtype=np.int)
        for start in start_position:
            assert start < seq_length
            start_label[start] = 1
        for end in end_position:
            assert end < seq_length
            end_label[end] = 1
        label_mask = np.ones([seq_length], dtype=np.int)
        return qas_id, query_token_ids, video_feature, text_feature, start_label, end_label, label_mask


class FeatureDataModule(pl.LightningDataModule):
    """
    加载提取好的视频文本特征
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    @staticmethod
    def load_examples(path):
        examples = []
        with open(path, "r", encoding="utf-8") as g:
            data = json.load(g)
            for index, d in enumerate(data):
                examples.append(InputExample(
                    qas_id=f"train-{index}",
                    video_name=d["video_name"],
                    query_text=d["query_text"],
                    video_feature_path=d["video_feature_path"],
                    text_feature_path=d["text_feature_path"],
                    length=d["length"],
                    start_position=d["start_position"],
                    end_position=d["end_position"]
                ))
        return examples

    def convert_single_example(self, example, max_length, mode="train"):
        sentence = example.query_text
        encoder = self.tokenizer(sentence, max_length=max_length, truncation=True)
        input_ids = encoder.input_ids
        segment_ids = encoder.token_type_ids
        input_mask = encoder.attention_mask
        if mode == "train":
            feature = InputFeature(
                qas_id=example.qas_id,
                video_name=example.video_name,
                video_feature_path=example.video_feature_path,
                text_feature_path=example.text_feature_path,
                length=example.length,
                query_token_ids=input_ids,
                start_position=example.start_position,
                end_position=example.end_position
            )
        else:
            feature = InputFeature(
                qas_id=example.qas_id,
                video_name=example.video_name,
                video_feature_path=example.video_feature_path,
                text_feature_path=example.text_feature_path,
                length=example.length,
                query_token_ids=input_ids
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

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            logger.info("loading train and val dataset")
            train_example = self.load_examples(self.args.train_data)
            train_feature = self.convert_examples_to_features(train_example, max_length=self.args.max_length)
            self.train_dataset = FeatureDataset(train_feature)
            val_example = self.load_examples(self.args.test_data)
            val_feature = self.convert_examples_to_features(val_example, max_length=self.args.max_length)
            self.val_dataset = FeatureDataset(val_feature)

    def _collate_fn(self, data):
        batch_id, batch_query_token_ids, batch_video_feature, batch_text_feature, batch_start_label, \
        batch_end_label, batch_label_mask = [], [], [], [], [], [], []
        for qas_id, query_token_ids, video_feature, text_feature, start_label, end_label, label_mask in data:
            batch_video_feature.append(video_feature)
            batch_text_feature.append(text_feature)
            batch_start_label.append(start_label)
            batch_end_label.append(end_label)
            batch_label_mask.append(label_mask)
            batch_query_token_ids.append(query_token_ids)
        # TODO pad video and text feature
        batch_video_feature = sequence_padding(batch_video_feature, seq_dims=2)
        batch_text_feature = sequence_padding(batch_text_feature, seq_dims=2)
        batch_query_token_ids = sequence_padding(batch_query_token_ids, value=self.tokenizer.pad_token_id)
        batch_label_mask = sequence_padding(batch_label_mask, value=0)
        batch_start_label = sequence_padding(batch_start_label, value=0)
        batch_end_label = sequence_padding(batch_end_label, value=0)
        return {"query_token_ids": convert_array_to_tensor(batch_query_token_ids),
                # 加载特征的tensor 类型需要转换
                "video_features": convert_array_to_tensor(batch_video_feature).float(),
                "text_features": convert_array_to_tensor(batch_text_feature).float(),
                "start_position": convert_array_to_tensor(batch_start_label),
                "end_position": convert_array_to_tensor(batch_end_label),
                "labels_mask": convert_array_to_tensor(batch_label_mask)
                }

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=self.args.num_workers,
                          collate_fn=self._collate_fn
                          )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=self.args.num_workers,
                          collate_fn=self._collate_fn
                          )


class VideoMomentLocBaseModel(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super(VideoMomentLocBaseModel, self).__init__()
        self.args = args
        self.bert_config = BertConfig.from_pretrained(self.args.bert_model)
        self.model = SimpleVideoMomentLoc(embedding_dim=self.args.embedding_size,
                                          hidden_size=self.args.hidden_size,
                                          num_layers=2,
                                          drop_rate=self.args.drop_rate,
                                          )
        self.bert = BertModel.from_pretrained(self.args.bert_model,
                                              config=self.bert_config
                                              )
        self.example_input_array = (torch.rand([16, 120, 2048]),
                                    torch.rand([16, 120, 768]),
                                    torch.rand([16, 768])
                                    )
        self.iou = mIOU()
        self.save_hyperparameters(args)

    @staticmethod
    def add_model_special_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("model hyper params")
        parser.add_argument("--embedding_size", type=int, default=2816,
                            help="video embedding size add text embedding size")
        parser.add_argument("--hidden_size", type=int, default=768, help="query embedding liner hidden size")
        parser.add_argument("--loss_type", type=str, choices=["ce", "bce", "focal_loss"], help="loss type")
        return parent_parser

    def forward(self, video_feature, text_feature, query_feature):
        logits = self.model(video_feature, text_feature, query_feature)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits

    def configure_optimizers(self):
        # 冻结bert
        for n, p in self.bert.named_parameters():
            p.requires_grad = False

        # 需要重新定义优化器和学习率
        no_decay = ['bias', 'layer_norm', 'LayerNorm']
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(
                [{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                  'weight_decay': self.args.weight_decay},
                 {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                  'weight_decay': 0.0}],
                lr=self.args.lr,
                # weight_decay=self.args.weight_decay,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                [{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                  'weight_decay': self.args.weight_decay},
                 {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                  'weight_decay': 0.0}],
                lr=self.args.lr,
                momentum=self.args.momentum,
            )
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels, label_mask):
        if self.args.loss_type == "bce":
            start_loss = F.binary_cross_entropy_with_logits(start_logits.view(-1), start_labels.view(-1).float(),
                                                            reduction="none")
            end_loss = F.binary_cross_entropy_with_logits(end_logits.view(-1), end_labels.view(-1).float(),
                                                          reduction="none")
            start_loss = (start_loss * label_mask.view(-1)).sum() / label_mask.sum()
            end_loss = (end_loss * label_mask.view(-1)).sum() / label_mask.sum()
        elif self.args.loss_type == "focal_loss":
            loss_fct = FocalLoss(gamma=self.args.focal_gamma, reduction="none")
            start_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(start_logits.view(-1)),
                                  start_labels.view(-1))
            end_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(end_logits.view(-1)),
                                end_labels.view(-1))
            start_loss = (start_loss * label_mask.view(-1)).sum() / label_mask.sum()
            end_loss = (end_loss * label_mask.view(-1)).sum() / label_mask.sum()

        else:
            raise ValueError("This type of loss func do not exists.")
        total_loss = (start_loss + end_loss) / 2

        return total_loss, start_loss, end_loss

    def on_train_start(self) -> None:
        self.logger.log_graph(self, self.example_input_array)
        logger.info("logging graph and start training model")

    def training_step(self, *args, **kwargs):
        batch, batch_idx = args
        video_feature = batch["video_features"]
        text_feature = batch["text_features"]
        start_labels = batch["start_position"]
        end_labels = batch["end_position"]
        query_token_ids = batch["query_token_ids"]
        label_mask = batch["labels_mask"]
        video_feature = torch.squeeze(video_feature, dim=1)
        text_feature = torch.squeeze(text_feature, dim=1)
        pooled_output = self.bert(query_token_ids).pooler_output
        start_logits, end_logits = self.forward(video_feature=video_feature,
                                                text_feature=text_feature,
                                                query_feature=pooled_output)
        total_loss, start_loss, end_loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels,
                                                             label_mask)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("train_start_loss", start_loss, prog_bar=True)
        self.log("train_end_loss", end_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self) -> None:
        for name, param in self.model.video_lstm.named_parameters():
            self.logger.experiment.add_histogram(name + "_grad", param.grad, global_step=self.global_step)

        for name, param in self.model.multimodal_lstm.named_parameters():
            self.logger.experiment.add_histogram(name + "_grad", param.grad, global_step=self.global_step)
        for name, param in self.model.linear.named_parameters():
            self.logger.experiment.add_histogram(name + "linear_grad", param.grad, global_step=self.global_step)

    def validation_step(self, *args, **kwargs):
        batch, batch_idx = args
        video_feature = batch["video_features"]
        text_feature = batch["text_features"]
        start_labels = batch["start_position"]
        end_labels = batch["end_position"]
        query_token_ids = batch["query_token_ids"]
        label_mask = batch["labels_mask"]
        video_feature = torch.squeeze(video_feature, dim=1)
        text_feature = torch.squeeze(text_feature, dim=1)
        pooled_output = self.bert(query_token_ids).pooler_output
        start_logits, end_logits = self.forward(video_feature=video_feature,
                                                text_feature=text_feature,
                                                query_feature=pooled_output)
        total_loss, start_loss, end_loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels,
                                                             label_mask)

        with torch.no_grad():
            target = torch.unsqueeze(start_labels, dim=-1).long() & torch.unsqueeze(end_labels, dim=1).long()
            target = [torch.nonzero(s.squeeze(0)) for s in torch.split(target, split_size_or_sections=1, dim=0)]
            pred_span, pred_span_score, pred_span_mask = self.extract_span(start_logits, end_logits, label_mask)
            preds = self.format_topk(pred_span, pred_span_score, pred_span_mask)
            target = torch.squeeze(torch.stack(target)).to(label_mask)
            preds = torch.squeeze(preds).to(label_mask)
            self.iou(preds, target)

        self.log("mIou", self.iou, prog_bar=True)
        self.log("val_loss", total_loss, prog_bar=True)

    @staticmethod
    def batch_gather(inputs, indices):
        """
        Gathers (batched) vectors according to indices.
        Arguments:
            inputs: Tensor[N, L, D]
            indices: Tensor[N, K] or Tensor[N]
        Returns:
            Tensor[N, K, D] or Tensor[N, D]
        """
        N, L, D = inputs.shape
        squeeze = False
        if indices.ndim == 1:
            squeeze = True
            indices = indices.unsqueeze(-1)
        N2, K = indices.shape
        assert N == N2
        indices = einops.repeat(indices, "N K -> N K D", D=D)
        out = torch.gather(inputs, dim=1, index=indices)
        if squeeze:
            out = out.squeeze(1)
        return out

    @staticmethod
    def format_topk(span, score, mask, topk=1):
        """
        https://discuss.pytorch.org/t/how-to-select-specific-vector-in-3d-tensor-beautifully/37724
        """
        assert topk >= 1, "topk value should larger than 1"
        batch_size = span.shape[0]
        num = span.shape[1]
        # 如果有结果
        out = []
        if num > 0:
            mask_score = torch.where(mask.bool(), score, torch.tensor(float('-inf')).to(score))
            value, index = torch.topk(mask_score, topk, dim=1, largest=True)
            for i, v, s in zip(index, value, span):
                topk_res = []
                for i_index, v_index in zip(i, v):
                    # 遍历topk结果
                    if not torch.isinf(v_index):
                        topk_res.append(torch.index_select(s, 0, i_index))
                    else:
                        topk_res.append(torch.zeros([1, 2]))
                out.append(torch.stack(topk_res))
            return torch.stack(out).to(span)
        else:
            return torch.zeros([batch_size, topk, 2]).to(span)

    @staticmethod
    def extract_span(start_logits, end_logits, label_mask, threshold=0.5):
        batch_size = start_logits.shape[0]
        start_score = torch.sigmoid(start_logits)
        end_score = torch.sigmoid(end_logits)
        score_matrix = torch.unsqueeze(start_score, dim=-1) + torch.transpose(torch.unsqueeze(end_score, dim=-1), 1, 2)
        # score matrix [batch_size,seq_length,seq_length]
        start_preds, end_preds = torch.sigmoid(start_logits) > threshold, torch.sigmoid(
            end_logits) > threshold
        # match_matrix = torch.zeros([batch_size, seq_length, seq_length])
        start_preds = start_preds.bool() & label_mask.bool()
        end_preds = end_preds.bool() & label_mask.bool()
        match_matrix = torch.unsqueeze(start_preds, dim=-1) & torch.transpose(torch.unsqueeze(end_preds, dim=-1), 1, 2)
        match_matrix = torch.triu(match_matrix, diagonal=0)
        match = torch.nonzero(match_matrix, as_tuple=False)
        score = score_matrix[list(match.T)]
        b, s, e = torch.split(match, dim=-1, split_size_or_sections=1)
        # b=batch_index,s=start_index,e=end_index
        # 构建基于batch的span mask
        index = torch.arange(batch_size).to(b).unsqueeze(-1).expand(-1, match.shape[0])
        span_mask = (torch.transpose(b, 0, 1).expand(batch_size, -1) == index)

        spans = torch.transpose(torch.stack([s, e], dim=-1), 0, 1).expand(batch_size, -1, -1)
        # 计算每个span的score
        spans_score = torch.unsqueeze(score, dim=0).expand(batch_size, -1)
        # match = [torch.nonzero(m.squeeze(0), as_tuple=False) for m in
        #          torch.split(match_matrix, split_size_or_sections=1, dim=0)]
        # score = [s.squeeze(0) for s in torch.split(score_matrix, split_size_or_sections=1, dim=0)]

        # spans = []
        # for index, (m, s) in enumerate(zip(match, score)):
        #     match_info = []
        #     if m.shape[0] > 0:
        #         for sm in torch.split(m, split_size_or_sections=1, dim=0):
        #             match_info.append([sm[0], sm[1], s[sm[0]]][sm[1]])
        #     spans.append(match_info)
        # span_topk = []
        # for i, span in enumerate(spans):
        #     # 先排序
        #     s = sorted(span, key=lambda x: x[2], reverse=True)
        #     if len(s) > topk:
        #         span_topk.append(s[:topk])
        #     else:
        #         left = topk - len(s)
        #         for i in range(left):
        #             s.append([])
        #         span_topk.append(s)
        return spans, spans_score, span_mask


def setup_seed(seed):
    pl.seed_everything(seed)
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    setup_seed(42)
    parser = argparse.ArgumentParser(description="hyper params")
    parser.add_argument("--batch_size", type=int, help="batch size in model")
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, help="default learning rate")
    parser.add_argument("--drop_rate", type=float, help="drop out rate")
    parser.add_argument("--optimizer_name", type=str, choices=["Adam", "SGD"], help="model optimizer")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=2e-4, type=float)
    parser.add_argument("--bert_model", type=str, help="bert model path for query feature extract")

    # =====================data params===========================
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--max_length", type=int, help="max query text length")

    # =======================Train params========================
    parser.add_argument("--early_stop", action="store_true", help="using early stop or not")
    parser.add_argument("--patience", type=int, default=10, help="early stop patience default is 10 epoch")
    parser = pl.Trainer.add_argparse_args(parser)

    # =======================Model params========================
    parser = VideoMomentLocBaseModel.add_model_special_args(parser)

    # ====================设置相关默认参数====================
    parser.set_defaults(
        optimizer_name="Adam",
        drop_rate=0.3
    )

    args = parser.parse_args()

    # ====================定制call backs=========================================
    call_backs = []
    checkpoint_callback = ModelCheckpoint(dirpath=args.default_root_dir,
                                          monitor='val_loss',
                                          filename='multimodal_video-{epoch:02d}-{val_loss:.2f}')
    if args.early_stop:
        early_stop = EarlyStopping("val_loss", mode="min", patience=args.patience, min_delta=0.001, verbose=True)
        call_backs.append(early_stop)

    call_backs.append(checkpoint_callback)
    call_backs.append(LearningRateMonitor(logging_interval="step"))
    # ========================================================================
    trainer = pl.Trainer.from_argparse_args(args, callbacks=call_backs)
    model = VideoMomentLocBaseModel(args)
    data_module = FeatureDataModule(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
