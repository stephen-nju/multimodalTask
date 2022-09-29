# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: train_swin_patch244_windows877.py
@time: 2022/9/26 15:14
"""

import argparse
import itertools
import logging
import torchmetrics
import os
from model.modeling_swin_transformer.video_swin_transformer import SwinTransformer3D
import pytorch_lightning as pl
import pytorchvideo.data
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

from torch.utils.data import DistributedSampler, RandomSampler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, args):
        """
        This LightningModule implementation constructs a PyTorchVideo ResNet,
        defines the train and val loss to be trained with (cross_entropy), and
        configures the optimizer.
        """
        self.args = args
        super().__init__()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.model = SwinTransformer3D()

    def forward(self, x):
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the training epoch. It must
        return a loss that is used for loss.backwards() internally. The self.log(...)
        function can be used to log any training metrics.
        PyTorchVideo batches are dictionaries containing each modality or metadata of
        the batch collated video clips. Kinetics contains the following notable keys:
           {
               'video': <video_tensor>,
               'audio': <audio_tensor>,
               'label': <action_label>,
           }
        - "video" is a Tensor of shape (batch, channels, time, height, Width)
        - "audio" is a Tensor of shape (batch, channels, time, 1, frequency)
        - "label" is a Tensor of shape (batch, 1)
        The PyTorchVideo models and transforms expect the same input shapes and
        dictionary structure making this function just a matter of unwrapping the dict and
        feeding it through the model/loss.
        """
        x = batch['video']
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        """
        We use the SGD optimizer with per step cosine annealing scheduler.
        """
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


class LiveVideoDataModule(pl.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

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

    def train_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        sampler = RandomSampler
        train_transform = self._make_transforms(mode="train")
        self.train_dataset = LimitDataset(
            pytorchvideo.data.Kinetics(
                data_path="D:\\Users\\Desktop\\train.txt",
                clip_sampler=pytorchvideo.data.make_clip_sampler(
                    "random", self.args.clip_duration
                ),
                video_path_prefix=self.args.video_path_prefix,
                transform=train_transform,
                video_sampler=sampler,
            )
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            # num_workers=self.args.workers,

        )


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


def main():
    """
    To train the ResNet with the Kinetics dataset we construct the two modules above,
    and pass them to the fit function of a pl.Trainer.
    This example can be run either locally (with default parameters) or on a Slurm
    cluster. To run on a Slurm cluster provide the --on_cluster argument.
    """
    setup_logger()

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
    parser.set_defaults(
        max_epochs=200,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
    )

    # Build trainer, ResNet lightning-module and Kinetics data-module.
    args = parser.parse_args()

    train(args)


def train(args):
    trainer = pl.Trainer.from_argparse_args(args)
    classification_module = VideoClassificationLightningModule(args)
    data_module = LiveVideoDataModule(args)
    trainer.fit(classification_module, data_module)


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


if __name__ == "__main__":
    main()
