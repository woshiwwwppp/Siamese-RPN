
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.ops import DeformConv
from lib.core import (
    AnchorGenerator, anchor_target, ga_loc_target, ga_shape_target, delta2bbox,
    multi_apply, weighted_sigmoid_focal_loss, weighted_binary_cross_entropy,
    weighted_bounded_iou_loss, multiclass_nms,weighted_cross_entropy,weighted_smoothl1)
from lib.utils import bias_init_with_prob
import numpy as np


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags



class SiamRPN(nn.Module):
    def __init__(self, size=1, feature_out=256, anchor=1):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x==3 else x*size, configs))
        feat_in = configs[-1]
        super(SiamRPN, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1] , kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.num_anchors = anchor
        self.anchor = anchor
        self.feature_out = feature_out
        self.use_sigmoid_cls=False
        if self.use_sigmoid_cls:
            self.cls_out_channels = 2 - 1
        else:
            self.cls_out_channels = 2
        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_loc1=nn.Conv2d(feat_in, feature_out*anchor, 3)
        self.conv_loc2=nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.octave_base_scale=4
        self.octave_ratios = [0.5, 1.0, 2.0]
        self.anchor_strides =list([2, 4, 8, 16, 32])
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)
        self.feat_channels=256
        self.deformable_groups=4
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 271,
                    'adaptive': False}  # 0.355
        self.anchor_base_sizes =  self.anchor_strides
        self.octave_scales = 8 * np.array(
            [2**(i / 3) for i in range(3)])
        self.approxs_per_octave = len(self.octave_scales) * len(self.octave_ratios)
        self.r1_kernel = []
        self.cls1_kernel = []
        self.loc1_kernel=[]
        self.approx_generators = []
        self.square_generators = []
        self.anchoring_means = (.0, .0, .0, .0)
        self.anchoring_stds = (1.0, 1.0, 1.0, 1.0)
        self.target_means = (.0, .0, .0, .0)
        self.target_stds = (1.0, 1.0, 1.0, 1.0)
        self.loc_filter_thr = 0.01
        self.cls_focal_loss = False
        self.loc_focal_loss=False
        for anchor_base in self.anchor_base_sizes:
            # Generators for approxs
            self.approx_generators.append(
                AnchorGenerator(anchor_base, self.octave_scales,
                                self.octave_ratios))
            # Generators for squares
            self.square_generators.append(
                AnchorGenerator(anchor_base, [self.octave_base_scale], [1.0]))
        self.feature_adaption = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)

        self.feature_adaption.init_weights()
    def forward(self, x,shapeAdaption):
        x_f = self.featureExtract(x)
        xx0=self.conv_loc2(x_f)
        xx1=self.conv_r2(x_f)
        xx1 = self.feature_adaption(xx1, shapeAdaption[0])
        xx2=self.conv_cls2(x_f)
        xx2 = self.feature_adaption(xx2, shapeAdaption[0])
        loc = list([F.conv2d(xx0, self.loc1_kernel)])
        bbox_pred=list([self.regress_adjust(F.conv2d(xx1, self.r1_kernel))])
        cls_score=list([F.conv2d(xx2, self.cls1_kernel)])
        return cls_score,bbox_pred,loc
        # # masked conv is only used during inference for speed-up
        # cls_score = self.conv_cls(x)
        # bbox_pred = self.conv_reg(x)

    def temple(self, z):
        z_f = self.featureExtract(z)
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        loc1_kernel_raw = self.conv_loc1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)
        self.loc1_kernel=loc1_kernel_raw.view(self.anchor, self.feature_out, kernel_size, kernel_size)
    def get_sampled_approxs(self, featmap_sizes, img_metas, cfg):
        """Get sampled approxs and inside flags according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: approxes of each image, inside flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # approxes for one time
        multi_level_approxs = []
        for i in range(num_levels):
            approxs = self.approx_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_approxs.append(approxs)
        approxs_list = [multi_level_approxs for _ in range(num_imgs)]

        # for each image, we compute inside flags of multi level approxes
        inside_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            multi_level_approxs = approxs_list[img_id]
            for i in range(num_levels):
                approxs = multi_level_approxs[i]
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.approx_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                inside_flags_list = []
                for i in range(self.approxs_per_octave):
                    split_valid_flags = flags[i::self.approxs_per_octave]
                    split_approxs = approxs[i::self.approxs_per_octave, :]
                    inside_flags = anchor_inside_flags(
                        split_approxs, split_valid_flags,
                        img_meta['img_shape'][:2], cfg['allowed_border'])
                    inside_flags_list.append(inside_flags)
                # inside_flag for a position is true if any anchor in this
                # position is true
                inside_flags = (torch.stack(inside_flags_list, 0).sum(dim=0) >
                                0)
                multi_level_flags.append(inside_flags)
            inside_flag_list.append(multi_level_flags)
        return approxs_list, inside_flag_list

    def get_anchors(self,
                    featmap_sizes,
                    shape_preds,
                    loc_preds,
                    img_metas,
                    use_loc_filter=False):
        """Get squares according to feature map sizes and guided
        anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            shape_preds (list[tensor]): Multi-level shape predictions.
            loc_preds (list[tensor]): Multi-level location predictions.
            img_metas (list[dict]): Image meta info.
            use_loc_filter (bool): Use loc filter or not.

        Returns:
            tuple: square approxs of each image, guided anchors of each image,
                loc masks of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # squares for one time
        multi_level_squares = []
        for i in range(num_levels):
            squares = self.square_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_squares.append(squares)
        squares_list = [multi_level_squares for _ in range(num_imgs)]

        # for each image, we compute multi level guided anchors
        guided_anchors_list = []
        loc_mask_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_guided_anchors = []
            multi_level_loc_mask = []
            for i in range(num_levels):
                squares = squares_list[img_id][i]
                shape_pred = shape_preds[i][img_id]
                loc_pred = loc_preds[i][img_id]
                guided_anchors, loc_mask = self.get_guided_anchors_single(
                    squares,
                    shape_pred,
                    loc_pred,
                    use_loc_filter=use_loc_filter)
                multi_level_guided_anchors.append(guided_anchors)
                multi_level_loc_mask.append(loc_mask)
            guided_anchors_list.append(multi_level_guided_anchors)
            loc_mask_list.append(multi_level_loc_mask)
        return squares_list, guided_anchors_list, loc_mask_list

    def get_guided_anchors_single(self,
                                  squares,
                                  shape_pred,
                                  loc_pred,
                                  use_loc_filter=False):
        """Get guided anchors and loc masks for a single level.

        Args:
            square (tensor): Squares of a single level.
            shape_pred (tensor): Shape predections of a single level.
            loc_pred (tensor): Loc predections of a single level.
            use_loc_filter (list[tensor]): Use loc filter or not.

        Returns:
            tuple: guided anchors, location masks
        """
        # calculate location filtering mask
        loc_pred = loc_pred.sigmoid().detach()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0.0
        mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_anchors)
        mask = mask.contiguous().view(-1)
        # calculate guided anchors
        squares = squares[mask]
        anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(
            -1, 2).detach()[mask]
        bbox_deltas = anchor_deltas.new_full(squares.size(), 0)
        bbox_deltas[:, 2:] = anchor_deltas
        guided_anchors = delta2bbox(squares,
                                    bbox_deltas,
                                    self.anchoring_means,
                                    self.anchoring_stds,
                                    wh_ratio_clip=1e-6)
        return guided_anchors, mask

    # def loss_shape_single(self, shape_pred, bbox_anchors, bbox_gts,
    #                       anchor_weights, anchor_total_num):
    #     shape_pred = shape_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
    #     bbox_anchors = bbox_anchors.contiguous().view(-1, 4)
    #     bbox_gts = bbox_gts.contiguous().view(-1, 4)
    #     anchor_weights = anchor_weights.contiguous().view(-1, 4)
    #     bbox_deltas = bbox_anchors.new_full(bbox_anchors.size(), 0)
    #     bbox_deltas[:, 2:] += shape_pred
    #     # filter out negative samples to speed-up weighted_bounded_iou_loss
    #     inds = torch.nonzero(anchor_weights[:, 0] > 0).squeeze(1)
    #     bbox_deltas_ = bbox_deltas[inds]
    #     bbox_anchors_ = bbox_anchors[inds]
    #     bbox_gts_ = bbox_gts[inds]
    #     anchor_weights_ = anchor_weights[inds]
    #     pred_anchors_ = delta2bbox(
    #         bbox_anchors_,
    #         bbox_deltas_,
    #         self.anchoring_means,
    #         self.anchoring_stds,
    #         wh_ratio_clip=1e-6)
    #     loss_shape = weighted_bounded_iou_loss(
    #         pred_anchors_,
    #         bbox_gts_,
    #         anchor_weights_,
    #         beta=0.2,
    #         avg_factor=anchor_total_num)
    #     return loss_shape

    def loss_loc_single(self, loc_pred, loc_target, loc_weight, loc_avg_factor,
                        cfg):
        if self.loc_focal_loss:
            loss_loc = weighted_sigmoid_focal_loss(
                loc_pred.reshape(-1, 1),
                loc_target.reshape(-1, 1).long(),
                loc_weight.reshape(-1, 1),
                avg_factor=loc_avg_factor)
        else:
            loss_loc = weighted_binary_cross_entropy(
                loc_pred.reshape(-1, 1),
                loc_target.reshape(-1, 1).long(),
                loc_weight.reshape(-1, 1),
                avg_factor=loc_avg_factor)
        if hasattr(cfg, 'loc_weight'):
            loss_loc = loss_loc * cfg.loc_weight
        return loss_loc


    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        if self.use_sigmoid_cls:
            if self.cls_focal_loss:
                cls_criterion = weighted_sigmoid_focal_loss
            else:
                cls_criterion = weighted_binary_cross_entropy
        else:
            if self.cls_focal_loss:
                raise NotImplementedError
            else:
                cls_criterion = weighted_cross_entropy
        if self.cls_focal_loss:
            loss_cls = cls_criterion(
                cls_score,
                labels,
                label_weights,
                gamma=cfg.gamma,
                alpha=cfg.alpha,
                avg_factor=num_total_samples)
        else:
            loss_cls = cls_criterion(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=1.0,
            avg_factor=num_total_samples)
        return loss_cls, loss_reg


    # def loss_single(self, cls_score, bbox_pred, labels, label_weights,
    #                 bbox_targets, bbox_weights, num_total_samples, cfg):
    #     # classification loss
    #     labels = labels.reshape(-1)
    #     label_weights = label_weights.reshape(-1)
    #     cls_score = cls_score.permute(0, 2, 3,
    #                                   1).reshape(-1, self.cls_out_channels)
    #     loss_cls = self.loss_cls(
    #         cls_score, labels, label_weights, avg_factor=num_total_samples)
    #     # regression loss
    #     bbox_targets = bbox_targets.reshape(-1, 4)
    #     bbox_weights = bbox_weights.reshape(-1, 4)
    #     bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    #     loss_bbox = self.loss_bbox(
    #         bbox_pred,
    #         bbox_targets,
    #         bbox_weights,
    #         avg_factor=num_total_samples)
    #     return loss_cls, loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             loc_preds,
             shape_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # assert len(featmap_sizes) == len(self.approx_generators)
        loc_targets, loc_weights, loc_avg_factor = ga_loc_target(
            gt_bboxes,
            featmap_sizes,
            self.octave_base_scale,
            self.anchor_strides,
            center_ratio=0.2,
            ignore_ratio=0.5)
        approxs_list, inside_flag_list = self.get_sampled_approxs(
            featmap_sizes, img_metas, cfg)

        squares_list, guided_anchors_list, _ = self.get_anchors(
            featmap_sizes, shape_preds, loc_preds, img_metas)

        # sampling = False if not hasattr(cfg, 'ga_sampler') else True
        # shape_targets = ga_shape_target(
        #     approxs_list,
        #     valid_flag_list,
        #     squares_list,
        #     gt_bboxes,
        #     img_metas,
        #     self.approxs_per_octave,
        #     cfg,
        #     sampling=sampling)
        # if shape_targets is None:
        #     return None
        # (bbox_anchors_list, bbox_gts_list, anchor_weights_list,
        #  all_inside_flags, anchor_fg_num, anchor_bg_num) = shape_targets
        # anchor_total_num = (anchor_fg_num
        #                     if not sampling else anchor_fg_num + anchor_bg_num)

        sampling = True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 2
        cls_reg_targets = anchor_target(
            guided_anchors_list,
            inside_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos if self.cls_focal_loss else
                             num_total_pos + num_total_neg)
        losses_cls, losses_reg = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        losses_loc, = multi_apply(
            self.loss_loc_single,
            loc_preds,
            loc_targets,
            loc_weights,
            loc_avg_factor=loc_avg_factor,
            cfg=cfg)
        # losses_shape, = multi_apply(
        #     self.loss_shape_single,
        #     shape_preds,
        #     bbox_anchors_list,
        #     bbox_gts_list,
        #     anchor_weights_list,
        #     anchor_total_num=anchor_total_num)
        Loss=losses_cls[0]+losses_reg[0]+losses_loc[0]
        return dict(
            Loss=Loss,
            loss_cls=losses_cls,
            loss_reg=losses_reg,
            # loss_shape=losses_shape,
            loss_loc=losses_loc)



    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   shape_preds,
                   loc_preds,
                   img_metas,
                   cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(shape_preds) == len(
            loc_preds)
        num_levels = len(cls_scores)
        mlvl_anchors = [
            self.square_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            shape_pred_list = [
                shape_preds[i][img_id].detach() for i in range(num_levels)
            ]
            loc_pred_list = [
                loc_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(
                cls_score_list, bbox_pred_list, shape_pred_list, loc_pred_list,
                mlvl_anchors, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          shape_preds,
                          loc_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, shape_pred, loc_pred, anchors in zip(
                cls_scores, bbox_preds, shape_preds, loc_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size(
            )[-2:] == shape_pred.size()[-2:] == loc_pred.size()[-2:]
            loc_pred = loc_pred.sigmoid()
            if not loc_pred.requires_grad:
                loc_mask = loc_pred >= self.loc_filter_thr
            else:
                loc_mask = loc_pred >= 0.0
            mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_anchors)
            mask = mask.contiguous().view(-1)
            mask_inds = mask.nonzero()
            if mask_inds.numel() == 0:
                continue
            else:
                mask_inds = mask_inds.squeeze()
            shape_pred = shape_pred.permute(1, 2, 0).reshape(-1, 2)
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = anchors[mask_inds, :]
            shape_pred = shape_pred[mask_inds, :]
            scores = scores[mask_inds, :]
            bbox_pred = bbox_pred[mask_inds, :]
            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                shape_pred = shape_pred.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                shape_pred = shape_pred[topk_inds, :]
            anchor_deltas = shape_pred.new_full((shape_pred.size(0), 4), 0)
            anchor_deltas[:, 2:] = shape_pred
            guided_anchors = delta2bbox(anchors, anchor_deltas,
                                        self.anchoring_means,
                                        self.anchoring_stds)
            bboxes = delta2bbox(guided_anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels


#
#
# class SiamRPNBIG(SiamRPN):
#     def __init__(self):
#         super(SiamRPNBIG, self).__init__(size=2)
#         self.cfg = {'lr':0.295, 'window_influence': 0.42, 'penalty_k': 0.055, 'instance_size': 271, 'adaptive': True} # 0.383
#
#
# class SiamRPNvot(SiamRPN):
#     def __init__(self):
#         super(SiamRPNvot, self).__init__(size=1, feature_out=256)
#         self.cfg = {'lr':0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 271, 'adaptive': False} # 0.355
#
#
# class SiamRPNotb(SiamRPN):
#     def __init__(self):
#         super(SiamRPNotb, self).__init__(size=1, feature_out=256)
#         self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'instance_size': 271, 'adaptive': False} # 0.655
#
#

class FeatureAdaption(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            2, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x


def normal_init(module, mean=0, std=1., bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

