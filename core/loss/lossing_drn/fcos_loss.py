# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: fcos_loss.py
@time: 2022/7/1 11:33
"""
import torch
import torch.nn as nn

from core.loss.iou_loss import IOULoss
from torchvision.ops import sigmoid_focal_loss

INF = 100000000


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.alpha = cfg["fcos_loss_alpha"]
        self.gamma = cfg["fcos_loss_gamma"]
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        # self.innerness_loss_func = nn.BCEWithLogitsLoss()
        self.iou_loss_func = nn.SmoothL1Loss()
        self.total_points = []

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 6],
            [5.6, 11],
            [11, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            # innerness_targets[i] = torch.split(innerness_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        # innerness_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        ts = locations
        # innerness_targets = []

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im * 32

            l = ts[:, None] - bboxes[None, 0]
            r = bboxes[None, 1] - ts[:, None]
            reg_targets_per_im = torch.cat([l, r], dim=1)

            is_in_boxes = reg_targets_per_im.min(dim=1)[0] > 0
            # innerness_targets.append(is_in_boxes)
            max_reg_targets_per_im = reg_targets_per_im.max(dim=1)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, 0]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, 1])

            locations_to_gt_area = bboxes[1] - bboxes[0]
            locations_to_gt_area = locations_to_gt_area.repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            labels_per_im = reg_targets_per_im.new_ones(len(reg_targets_per_im))
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def __call__(self, locations, box_cls, box_regression, targets, iou_scores, is_first_stage=True):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)
        # labels=[64,32,16]
        # reg_targets=[[64，2]，[32,2],[16,2]]
        box_cls_flatten = []
        box_regression_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 1).reshape(-1, 2))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 2))

        if not is_first_stage:
            # box_regression=[[batch,2,32],[batch,2,16],[batch,2,8]]
            merged_box_regression = torch.cat(box_regression, dim=-1).transpose(2, 1)
            # merged_box_regression=[batch,56,2]
            merged_locations = torch.cat(locations, dim=0)
            # merged_location=[56]
            full_locations = merged_locations[None, :].expand(merged_box_regression.size(0), -1).contiguous()
            pred_start = full_locations - merged_box_regression[:, :, 0]
            pred_end = full_locations + merged_box_regression[:, :, 1]
            # prediction=[batch, 56, 2]
            predictions = torch.true_divide(torch.cat([pred_start.unsqueeze(-1), pred_end.unsqueeze(-1)], dim=-1), 32)
            # TODO: make sure the predictions are legal. (e.g. start < end)
            predictions[:, 0].clamp_(min=0, max=1)
            predictions[:, 0].clamp_(min=0, max=1)
            # gt: [batch, 2]
            gt_box = targets[:, None, :]
            # TODO: double check here
            iou_target = segment_tiou(predictions, gt_box)
            iou_pred = torch.cat(iou_scores, dim=-1).squeeze().sigmoid()
            # TODO: select the predictions with iou > 0.5? for computing loss
            iou_pos_ind = iou_target > 0.9
            # print(f'Positive sample num: {iou_pos_ind.sum()}')

            pos_iou_target = iou_target[iou_pos_ind]
            pos_iou_pred = iou_pred[iou_pos_ind]
            self.total_points.append((pos_iou_pred, pos_iou_target))
            if iou_pos_ind.sum().item() == 0:
                iou_loss = torch.tensor([0.])
            else:
                iou_loss = self.iou_loss_func(pos_iou_pred, pos_iou_target)

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)

        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = sigmoid_focal_loss(
            box_cls_flatten,
            labels_flatten.int(),
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="sum"
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        # centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            # centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                # centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()

        if not is_first_stage:
            return cls_loss, reg_loss, iou_loss

        return cls_loss, reg_loss, torch.FloatTensor([0.]).type_as(cls_loss)


def segment_tiou(box_a, box_b):
    # gt: [batch, 1, 2], detections: [batch, 56, 2]
    # calculate interaction
    inter_max_xy = torch.min(box_a[:, :, -1], box_b[:, :, -1])
    inter_min_xy = torch.max(box_a[:, :, 0], box_b[:, :, 0])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

    # calculate union
    union_max_xy = torch.max(box_a[:, :, -1], box_b[:, :, -1])
    union_min_xy = torch.min(box_a[:, :, 0], box_b[:, :, 0])
    union = torch.clamp((union_max_xy - union_min_xy), min=0)

    iou = torch.true_divide(inter, (union + 1e-6))

    return iou


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
