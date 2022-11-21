# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: video_moment_loc_iou.py
@time: 2022/11/14 11:17
"""
from typing import Any
import torch
from torch import Tensor
from torchmetrics import Metric


def calculate_iou_accuracy(ious: Tensor, threshold: Tensor):
    size = ious.shape[0]
    count = (ious > threshold).long().sum()
    return count / size * 100


def iou(preds: Tensor, target: Tensor) -> Tensor:
    """
    https://github.com/Lightning-AI/lightning-bolts/blob/0.5.0/pl_bolts/metrics/object_detection.py
        preds=torch.tensor([[1,2],[3,4]])
        target=torch.tensor([[1,2],[1,2]])
    """

    intersection_start = torch.max(preds[:, 0], target[:, 0])
    intersection_end = torch.min(preds[:, 1], target[:, 1])
    intersection = (intersection_end - intersection_start).clamp(min=0)
    pred_area = (preds[:, 1] - preds[:, 0])
    target_area = (target[:, 1] - target[:, 0])
    union = pred_area + target_area - intersection
    iou = torch.true_divide(intersection, union)
    return iou


class mIOU(Metric):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("mIou", torch.zeros(1, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        target shape=[batch_size,2]
        preds shape=[batch_size,2]
        """
        ious = iou(preds=preds, target=target)
        mIou = torch.mean(ious)
        self.mIou += mIou
        self.total += 1

    def compute(self) -> Any:
        mIou = self.mIou.sum().float()
        mIou = mIou.float() / self.total
        return mIou


class RetrieveTopkIou(Metric):
    def __init__(self, topk=1, m=0.3, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("topkIou", torch.zeros(1, dtype=torch.double), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        pass

    def compute(self) -> Any:
        pass

# def update(self, start_logits, end_logits, start_target, end_target, label_mask) -> None:
#     start_preds, end_preds = torch.sigmoid(start_logits) > self.threshold, torch.sigmoid(
#         end_logits) > self.threshold
#     start_preds = start_preds.bool() & label_mask.bool()
#     end_preds = end_preds.bool() & label_mask.bool()
#     # start_positions = torch.nonzero(start_preds)
#     # end_positions = torch.nonzero(end_preds)
#     ious = []
#     for start, end, start_gt, end_gt in zip(start_preds, end_preds, start_target, end_target):
#         start_index = [idx for idx, tmp in enumerate(start) if tmp]
#         end_index = [idx for idx, tmp in enumerate(end) if tmp]
#         start_tg_index = [idx for idx, tmp in enumerate(start_gt) if tmp]
#         end_tg_index = [idx for idx, tmp in enumerate(end_gt) if tmp]
#         # TODO 可以筛选概率最大的区间，现在按规则选取一个最近的 end
#         if len(start_index) > 0 and len(end_index) > 0:
#             start_choice = start_index[0]
#             tmp_ends = [tmp for tmp in end_index if tmp >= start_choice]
#             if len(tmp_ends) > 0:
#                 end_choice = tmp_ends[0]
#                 iou = calculate_iou(i0=[start_choice, end_choice], i1=[start_tg_index[0], end_tg_index[0]])
#                 ious.append(iou)
#             else:
#                 ious.append(0)
#         if len(ious) > 0:
#             r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
#             r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
#             r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
#             mean = np.mean(ious) * 100.0
#             self.mIou = torch.tensor(mean)
#         else:
#             self.mIou += 0
