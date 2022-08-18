# -*- coding: utf-8 -*-
"""
@author: zhubin
@license: Apache Licence
@software:PyCharm
@file: fcos.py
@time: 2022/7/1 11:15
"""

import torch
import torch.nn as nn
import pickle
import math

from core.loss.lossing_drn.fcos_loss import make_fcos_loss_evaluator


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(
            self,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            min_size,
            num_classes,
            is_first_stage,
            is_second_stage
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.innerness_threshold = 0.15
        self.downsample_scale = 32
        self.is_first_stage = is_first_stage
        self.is_second_stage = is_second_stage

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, level, iou_scores
    ):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, T = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.permute(0, 2, 1).contiguous().sigmoid()
        iou_scores = iou_scores.permute(0, 2, 1).contiguous().sigmoid()
        box_regression = box_regression.permute(0, 2, 1)

        # centerness = centerness.permute(0, 2, 1)
        # centerness = centerness.reshape(N, -1).sigmoid()
        # inner = inner.squeeze().sigmoid()

        candidate_inds = (box_cls > self.pre_nms_thresh)
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        # box_cls = box_cls * centerness[:, :, None]
        # box_cls = box_cls + centerness[:, :, None]
        if not self.is_first_stage:
            box_cls = box_cls * iou_scores

        results = []
        for i in range(N):

            # per_centerness = centerness[i]

            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            # per_centerness = per_centerness[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

                # per_centerness = per_centerness[top_k_indices]

            detections = torch.stack([
                per_locations - per_box_regression[:, 0],
                per_locations + per_box_regression[:, 1],
            ], dim=1) / self.downsample_scale

            detections[:, 0].clamp_(min=0, max=1)
            detections[:, 1].clamp_(min=0, max=1)

            # remove small boxes
            p_start, p_end = detections.unbind(dim=1)
            duration = p_end - p_start
            keep = (duration >= self.min_size).nonzero().squeeze(1)
            detections = detections[keep]

            temp_dict = {}
            temp_dict["detections"] = detections
            temp_dict['labels'] = per_class
            temp_dict['scores'] = torch.sqrt(per_box_cls)
            temp_dict['level'] = [level]
            # temp_dict['centerness'] = per_centerness
            temp_dict['locations'] = per_locations / 32

            results.append(temp_dict)

        return results

    def forward(self, locations, box_cls, box_regression, iou_scores):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for i, (l, o, b, iou_s) in enumerate(zip(locations, box_cls, box_regression, iou_scores)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, i, iou_s
                )
            )

        boxlists = list(zip(*sampled_boxes))
        # boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            dicts = boxlists[i]
            per_vid_scores = []
            per_vid_detections = []
            per_vid_labels = []
            # add level number
            per_vid_level = []
            per_vid_locations = []
            per_vid_centerness = []
            for per_scale_dict in dicts:
                if len(per_scale_dict["detections"]) != 0:
                    per_vid_detections.append(per_scale_dict["detections"])
                if len(per_scale_dict["scores"]) != 0:
                    per_vid_scores.append(per_scale_dict["scores"])
                if len(per_scale_dict['level']) != 0:
                    per_vid_level.append(per_scale_dict['level'] * len(per_scale_dict['detections']))

                if len(per_scale_dict['locations']) != 0:
                    per_vid_locations.append(per_scale_dict['locations'])

                # if len(per_scale_dict['centerness']) != 0:
                #     per_vid_centerness.append(per_scale_dict['centerness'])
            if len(per_vid_detections) == 0:
                per_vid_detections = torch.Tensor([0, 1]).unsqueeze(0).cuda()
                per_vid_scores = torch.Tensor([1]).cuda()
                per_vid_level = [[-1]]
                per_vid_locations = torch.Tensor([0.5]).cuda()
                # per_vid_centerness = torch.Tensor([0.5]).cuda()
            else:
                per_vid_detections = torch.cat(per_vid_detections, dim=0)
                per_vid_scores = torch.cat(per_vid_scores, dim=0)
                per_vid_level = per_vid_level
                per_vid_locations = torch.cat(per_vid_locations, dim=0)
                # per_vid_centerness = torch.cat(per_vid_centerness, dim=0)

            temp_dict = {}
            temp_dict["detections"] = per_vid_detections
            temp_dict['labels'] = per_vid_labels
            temp_dict['scores'] = per_vid_scores
            temp_dict['level'] = per_vid_level
            # temp_dict['centerness'] = per_vid_centerness
            temp_dict['locations'] = per_vid_locations
            results.append(temp_dict)

        return results


def make_fcos_postprocessor(config):
    pre_nms_thresh = config["fcos_inference_thr"]
    pre_nms_top_n = config["fcos_pre_nms_top_n"]
    nms_thresh = config["fcos_nms_thr"]
    fpn_post_nms_top_n = config["test_detections_per_img"]
    is_first_stage = config['is_first_stage']
    is_second_stage = config['is_second_stage']
    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config["fcos_num_class"],
        is_first_stage=is_first_stage,
        is_second_stage=is_second_stage
    )

    return box_selector


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg["fcos_num_class"] - 1

        cls_tower = []
        bbox_tower = []
        for i in range(cfg["fcos_conv_layers"]):
            cls_tower.append(nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm1d(in_channels))
            cls_tower.append(nn.ReLU())
            # cls_tower.append((nn.Dropout(p=0.5)))
            bbox_tower.append(nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_tower.append(nn.BatchNorm1d(in_channels))
            bbox_tower.append(nn.ReLU())
            # bbox_tower.append((nn.Dropout(p=0.5)))

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv1d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )

        self.bbox_pred = nn.Conv1d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        )

        self.centerness = nn.Conv1d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        self.mix_fc = nn.Sequential(
            nn.Conv1d(2 * in_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        self.iou_scores = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, 1, kernel_size=1, stride=1),
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness, self.iou_scores, self.mix_fc]:
            for l in modules.modules():
                if isinstance(l, nn.Conv1d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg["fcos_prior_prob"]
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(3)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        iou_scores = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            logits.append(self.cls_logits(cls_tower))

            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(box_tower)
            )))
            mix_feature = self.mix_fc(torch.cat([cls_tower, box_tower], dim=1))
            iou_scores.append(self.iou_scores(mix_feature))

        return logits, bbox_reg, centerness, iou_scores


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()
        cfg = vars(cfg)
        head = FCOSHead(cfg, in_channels)
        self.is_first_stage = cfg['is_first_stage']
        box_selector_test = make_fcos_postprocessor(cfg)
        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg["fpn_stride"]

    def forward(self, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness, iou_scores = self.head(features)
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                targets, iou_scores
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                targets, iou_scores
            )

    def _forward_train(self, locations, box_cls, box_regression,
                       targets, iou_scores):
        loss_box_cls, loss_box_reg, loss_iou = self.loss_evaluator(
            locations, box_cls, box_regression, targets, iou_scores, self.is_first_stage
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            'loss_iou': loss_iou,
            # "loss_centerness": loss_centerness,
            # 'loss_innerness': loss_innerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression,
                      targets, iou_scores):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, iou_scores

        )

        loss_box_cls, loss_box_reg, loss_iou = self.loss_evaluator(
            locations, box_cls, box_regression, targets, iou_scores, self.is_first_stage
        )
        pickle.dump(self.loss_evaluator.total_points, open('total_points.pkl', 'wb'))

        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            'loss_iou': loss_iou
            # "loss_centerness": loss_centerness,
            # 'loss_innerness': loss_innerness
        }
        return boxes, losses

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            t = feature.size(-1)
            locations_per_level = self.compute_locations_per_level(
                t, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, t, stride, device):
        shifts_t = torch.arange(
            0, t * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_t = shifts_t.reshape(-1)
        locations = shifts_t + stride / 2
        return locations
