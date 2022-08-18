# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: train_drn.py
@time: 2022/6/30 15:06
"""
import json
import os
import numpy as np
import pytorch_lightning as pl
import argparse

import torch
from torch.utils.data import DataLoader
from loguru import logger
from model.modeling_drn.dense_regression_net import DenseRegressionNet
from pytorch_lightning import Trainer
from task.dense_regression_network.dataset import CharadesSTA, collate_data
from pytorch_lightning.callbacks import LearningRateMonitor


class DenseRegressModule(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.drn_model = None
        self.args = args
        self.init_model(args)
        self.save_hyperparameters()

    def init_model(self, args):
        # 加载词典
        word2idx = json.load(
            open('task/dense_regression_network/data/dataset/Charades/Charades_word2id.json', 'r'))
        self.drn_model = DenseRegressionNet(args, vocab_size=len(word2idx))
        if os.path.exists(self.args.glove_weights):
            logger.info("Loading glove weights")
            self.drn_model.query_encoder.embedding.weight.data.copy_(torch.load(args.glove_weights))
        else:
            # logger.info("Generating glove weights")
            self.drn_model.query_encoder.embedding.weight.data.copy_(self.glove_init(word2idx))

    def glove_init(self, word2id):
        # 加载原始词向量
        glove = list(open('task/dense_regression_network/data/glove_weights', 'r'))
        full_glove = {}
        for line in glove:
            values = line.strip('\n').split(" ")
            word = values[0]
            vector = np.asarray([float(e) for e in values[1:]])
            full_glove[word] = vector
        full_glove_keys = list(full_glove.keys())
        weight = torch.zeros((len(word2id) + 1, 300), dtype=torch.float)
        for word, idx in word2id.items():
            if word in list(full_glove_keys):
                glove_vector = torch.from_numpy(full_glove[word])
                weight[idx] = glove_vector
        torch.save(weight, self.args.glove_weights)

        return weight

    @staticmethod
    def add_model_specific_args(parent_parser):
        # 添加构建模型的参数
        model_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        fcos_parser = model_parser.add_argument_group("fcos config")
        fcos_parser.add_argument("--fcos_conv_layers", type=int, default=1)
        fcos_parser.add_argument("--fcos_prior_prob", type=float, default=0.01)
        fcos_parser.add_argument("--fcos_loss_alpha", type=float, default=0.25)
        fcos_parser.add_argument("--fcos_loss_gamma", type=float, default=2.0)
        fcos_parser.add_argument("--fcos_inference_thr", type=float, default=0.05)
        fcos_parser.add_argument("--fcos_pre_nms_top_n", type=int, default=32)
        fcos_parser.add_argument("--fcos_nms_thr", type=float, default=0.6)
        fcos_parser.add_argument("--fcos_num_class", type=int, default=2)
        fcos_parser.add_argument("--test_detections_per_img", type=int, default=32)
        backbone_parser = model_parser.add_argument_group("back bone")
        backbone_parser.add_argument("--first_output_dim", type=int, default=256)
        fpn_parser = model_parser.add_argument_group("fpn")
        fpn_parser.add_argument("--fpn_feature_dim", type=int, default=512)
        fpn_parser.add_argument("--fpn_stride", type=list, default=[1, 2, 4])

        return model_parser

    def train_dataloader(self):
        train_dataset = CharadesSTA(self.args, split='train')

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size,
            shuffle=True, collate_fn=collate_data, num_workers=8, pin_memory=True
        )
        return train_dataloader

    def training_step(self, *args, **kwargs):
        batch = args[0]
        batch_idx = args[1]
        vid_names, props_start_end, props_features, gt_start_end, query_tokens, query_len, props_num, num_frames = batch

        _, loss_dict = self.drn_model(
            query_tokens, query_len, props_features, props_start_end, gt_start_end, props_num, num_frames
        )

        if self.args.is_second_stage:
            loss = loss_dict['loss_iou']
        else:
            # loss = torch.sum(torch.stack([loss for loss in loss_dict.values()]))
            loss = sum(loss for loss in loss_dict.values())

        self.log("learning rate", self.trainer.optimizers[0].state_dict()['param_groups'][0]['lr'], prog_bar=True)
        return loss

    def configure_optimizers(self):
        learned_params = None
        if self.args.is_first_stage:
            for name, value in self.drn_model.named_parameters():
                if 'iou_scores' in name or 'mix_fc' in name:
                    value.requires_grad = False
            learned_params = filter(lambda p: p.requires_grad, self.drn_model.parameters())
        elif self.args.is_second_stage:
            head_params = self.drn_model.fcos.head.iou_scores.parameters()
            fc_params = self.drn_model.fcos.head.mix_fc.parameters()
            learned_params = list(head_params) + list(fc_params)
            self.args.lr /= 100
        elif self.args.is_third_stage:
            learned_params = self.drn_model.parameters()
            self.args.lr /= 10000

        optimizer = torch.optim.Adam(learned_params, self.args.lr)

        return [optimizer]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Language Driven Video Temporal Localization implemented in pyTorch")

    # =========================== Data Configs ==============================
    parser.add_argument('--dataset', type=str, default='Charades')
    parser.add_argument('--props_file_path', type=str,
                        default="task/dense_regression_network/data/dataset/Charades/Charades_MAN_32_props.txt")
    parser.add_argument('--feature_type', choices=['C3D', 'I3D', 'MFnet'], default="C3D")
    parser.add_argument("--feature_root", type=str)
    parser.add_argument("--feature_dim", type=int)
    parser.add_argument("--ft_window_size", type=int)
    parser.add_argument("--ft_overlap", type=float)
    # ===========================Logger configs ===============================
    parser.add_argument("--logger_dir", type=str, default="./output/logger")
    # =========================== Learning Configs ============================
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('--is_first_stage', action='store_true', )
    parser.add_argument('--is_second_stage', action='store_true')
    parser.add_argument('--is_third_stage', action='store_true')
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--snapshot_pref', type=str)
    parser.add_argument('--glove_weights', type=str, default="task/dense_regression_network/data/glove_weights")
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--lstm_layers', type=int, default=1)
    parser.add_argument('--node_ft_dim', type=int, default=1024)
    parser.add_argument('--graph_num_layers', type=int, default=3)
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--clip_gradient', type=float, default=0.5)
    parser.add_argument('--loss_weights', type=float, default=0.5)
    parser.add_argument('--loss_type', choices=['iou', 'bce'], default="iou")
    parser.add_argument('--start_epoch', type=int, default=0)

    parser.add_argument('--weight_decay', '--wd', type=float,
                        metavar='W', help='weight decay (default: 5e-4)')

    parser = DenseRegressModule.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    arg = parser.parse_args()

    # =============================添加callback======================================
    call_backs = []
    lr_monitor = LearningRateMonitor(logging_interval='step')
    call_backs.append(lr_monitor)

    trainer = Trainer.from_argparse_args(arg, callbacks=call_backs)
    # 模型构建
    model = DenseRegressModule(args=arg)
    if arg.resume:
        if os.path.isfile(arg.resume):
            model = DenseRegressModule.load_from_checkpoint(arg.resume, args=arg)
            logger.info(("=> loading checkpoint '{}'".format(arg.resume)))
            checkpoint = torch.load(arg.resume)
            arg.start_epoch = checkpoint['epoch']
            pretrained_dict = checkpoint['state_dict']
            # only resume part of model paramete
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # main_model.load_state_dict(checkpoint['state_dict'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})"
                         .format(arg.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(arg.resume)))

    # ==============================训练模型=========================================
    trainer.fit(model=model)

    # ============================测试dataset================================================
    # data = CharadesSTA(arg, split="train")
    # dl = DataLoader(data, batch_size=arg.batch_size,
    #                 shuffle=True, collate_fn=collate_data, num_workers=2, pin_memory=True)
    #
    # for d in dl:
    #     print(d)
