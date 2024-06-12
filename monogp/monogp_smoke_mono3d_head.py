from typing import List, Optional, Tuple

import torch
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models import SMOKEMono3DHead
from mmdet3d.registry import MODELS
from mmdet3d.structures import points_cam2img
from mmdet3d.utils import InstanceList


@MODELS.register_module()
class MonoGpSMOKEMono3DHead(SMOKEMono3DHead):

    def __init__(self,
                 *args,
                 use_ground_plane: bool = False,
                 pred_shift_height: bool = False,
                 origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 use_gravity_center_target: bool = False,
                 **kwargs) -> None:
        self.use_ground_plane = use_ground_plane
        self.pred_shift_height = pred_shift_height
        self.origin = origin
        self.use_gravity_center_target = use_gravity_center_target
        super().__init__(*args, **kwargs)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        cls_score, bbox_pred = super().forward_single(x)
        if self.pred_shift_height:
            # (N, C, H, W)
            shift_heights = bbox_pred[:, 8, ...]
            bbox_pred[:, 8, ...] = shift_heights.sigmoid() - 0.5
        return cls_score, bbox_pred

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        rescale: bool = None) -> InstanceList:
        assert len(cls_scores) == len(bbox_preds) == 1
        cam2imgs = torch.stack([
            cls_scores[0].new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas
        ])
        trans_mats = torch.stack([
            cls_scores[0].new_tensor(img_meta['trans_mat'])
            for img_meta in batch_img_metas
        ])
        planes = None
        if self.use_ground_plane:
            planes = torch.stack([
                cls_scores[0].new_tensor(img_meta['plane'])
                for img_meta in batch_img_metas
            ])
        batch_bboxes, batch_scores, batch_topk_labels = self._decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            batch_img_metas,
            cam2imgs=cam2imgs,
            trans_mats=trans_mats,
            planes=planes,
            topk=100,
            kernel=3)

        result_list = []
        for img_id in range(len(batch_img_metas)):

            bboxes = batch_bboxes[img_id]
            scores = batch_scores[img_id]
            labels = batch_topk_labels[img_id]

            keep_idx = scores > 0.25
            bboxes = bboxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]

            bboxes = batch_img_metas[img_id]['box_type_3d'](
                bboxes, box_dim=self.bbox_code_size, origin=self.origin)
            attrs = None

            results = InstanceData()
            results.bboxes_3d = bboxes
            results.labels_3d = labels
            results.scores_3d = scores

            if attrs is not None:
                results.attr_labels = attrs

            result_list.append(results)

        return result_list

    def _decode_heatmap(self,
                        cls_score: Tensor,
                        reg_pred: Tensor,
                        batch_img_metas: List[dict],
                        cam2imgs: Tensor,
                        trans_mats: Tensor,
                        planes: Optional[Tensor],
                        topk: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor, Tensor]:
        img_h, img_w = batch_img_metas[0]['pad_shape'][:2]
        bs, cs, feat_h, feat_w = reg_pred.shape

        center_heatmap_pred = get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = transpose_and_gather_feat(reg_pred, batch_index)
        regression = regression.view(-1, cs)

        points = torch.cat([topk_xs.view(-1, 1),
                            topk_ys.view(-1, 1).float()],
                           dim=1)
        locations, dimensions, orientations = self.bbox_coder.decode(
            regression, points, batch_topk_labels, cam2imgs, trans_mats,
            planes, self.use_ground_plane, self.pred_shift_height, self.origin)

        batch_bboxes = torch.cat((locations, dimensions, orientations), dim=1)
        batch_bboxes = batch_bboxes.view(bs, -1, self.bbox_code_size)
        return batch_bboxes, batch_scores, batch_topk_labels

    def get_predictions(self, labels_3d: Tensor, centers_2d: Tensor,
                        gt_locations: Tensor, gt_dimensions: Tensor,
                        gt_orientations: Tensor, indices: Tensor,
                        batch_img_metas: List[dict], pred_reg: Tensor) -> dict:
        batch, channel = pred_reg.shape[0], pred_reg.shape[1]
        w = pred_reg.shape[3]
        cam2imgs = torch.stack([
            gt_locations.new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas
        ])
        trans_mats = torch.stack([
            gt_locations.new_tensor(img_meta['trans_mat'])
            for img_meta in batch_img_metas
        ])
        planes = None
        if self.use_ground_plane:
            planes = torch.stack([
                gt_locations.new_tensor(img_meta['plane'])
                for img_meta in batch_img_metas
            ])
        centers_2d_inds = centers_2d[:, 1] * w + centers_2d[:, 0]
        centers_2d_inds = centers_2d_inds.view(batch, -1)
        pred_regression = transpose_and_gather_feat(pred_reg, centers_2d_inds)
        pred_regression_pois = pred_regression.view(-1, channel)
        locations, dimensions, orientations = self.bbox_coder.decode(
            pred_regression_pois, centers_2d, labels_3d, cam2imgs, trans_mats,
            planes, self.use_ground_plane, self.pred_shift_height, self.origin,
            gt_locations)

        locations, dimensions, orientations = locations[indices], dimensions[
            indices], orientations[indices]

        dst = locations.new_tensor((0.5, 1.0, 0.5))
        src = locations.new_tensor(self.origin)
        locations += dimensions * (dst - src)

        gt_locations = gt_locations[indices]

        assert len(locations) == len(gt_locations)
        assert len(dimensions) == len(gt_dimensions)
        assert len(orientations) == len(gt_orientations)
        bbox3d_yaws = self.bbox_coder.encode(gt_locations, gt_dimensions,
                                             orientations, batch_img_metas)
        bbox3d_dims = self.bbox_coder.encode(gt_locations, dimensions,
                                             gt_orientations, batch_img_metas)
        bbox3d_locs = self.bbox_coder.encode(locations, gt_dimensions,
                                             gt_orientations, batch_img_metas)

        pred_bboxes = dict(ori=bbox3d_yaws, dim=bbox3d_dims, loc=bbox3d_locs)

        return pred_bboxes

    def get_targets(self, batch_gt_instances_3d: InstanceList,
                    batch_gt_instances: InstanceList, feat_shape: Tuple[int],
                    batch_img_metas: List[dict]) -> Tuple[Tensor, int, dict]:
        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        gt_bboxes_3d = [
            gt_instances_3d.bboxes_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        gt_labels_3d = [
            gt_instances_3d.labels_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        centers_2d = [
            gt_instances_3d.centers_2d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        img_shape = batch_img_metas[0]['pad_shape']

        reg_mask = torch.stack([
            gt_bboxes[0].new_tensor(
                not img_meta['affine_aug'], dtype=torch.bool)
            for img_meta in batch_img_metas
        ])
        cam2imgs = torch.stack([
            gt_bboxes[0].new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas
        ])

        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)  # 1/4
        height_ratio = float(feat_h / img_h)  # 1/4

        assert width_ratio == height_ratio

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])

        gt_centers_2d = centers_2d.copy()
        if self.use_gravity_center_target:
            gt_centers_2d = [
                points_cam2img(gt_bbox_3d.gravity_center, cam2img)
                for gt_bbox_3d, cam2img in zip(gt_bboxes_3d, cam2imgs)
            ]

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            # project centers_2d from input image to feat map
            gt_center_2d = gt_centers_2d[batch_id] * width_ratio

            for j, center in enumerate(gt_center_2d):
                center_x_int, center_y_int = center.int()
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.7)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [center_x_int, center_y_int], radius)

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        num_ctrs = [center_2d.shape[0] for center_2d in centers_2d]
        max_objs = max(num_ctrs)

        reg_inds = torch.cat(
            [reg_mask[i].repeat(num_ctrs[i]) for i in range(bs)])

        inds = torch.zeros((bs, max_objs),
                           dtype=torch.bool).to(centers_2d[0].device)

        # put gt 3d bboxes to gpu
        gt_bboxes_3d = [
            gt_bbox_3d.to(centers_2d[0].device) for gt_bbox_3d in gt_bboxes_3d
        ]

        batch_centers_2d = centers_2d[0].new_zeros((bs, max_objs, 2))
        batch_labels_3d = gt_labels_3d[0].new_zeros((bs, max_objs))
        batch_gt_locations = \
            gt_bboxes_3d[0].tensor.new_zeros((bs, max_objs, 3))
        for i in range(bs):
            inds[i, :num_ctrs[i]] = 1
            batch_centers_2d[i, :num_ctrs[i]] = centers_2d[i]
            batch_labels_3d[i, :num_ctrs[i]] = gt_labels_3d[i]
            batch_gt_locations[i, :num_ctrs[i]] = \
                gt_bboxes_3d[i].tensor[:, :3]

        inds = inds.flatten()
        batch_centers_2d = batch_centers_2d.view(-1, 2) * width_ratio
        batch_gt_locations = batch_gt_locations.view(-1, 3)

        # filter the empty image, without gt_bboxes_3d
        gt_bboxes_3d = [
            gt_bbox_3d for gt_bbox_3d in gt_bboxes_3d
            if gt_bbox_3d.tensor.shape[0] > 0
        ]

        gt_dimensions = torch.cat(
            [gt_bbox_3d.tensor[:, 3:6] for gt_bbox_3d in gt_bboxes_3d])
        gt_orientations = torch.cat([
            gt_bbox_3d.tensor[:, 6].unsqueeze(-1)
            for gt_bbox_3d in gt_bboxes_3d
        ])
        gt_corners = torch.cat(
            [gt_bbox_3d.corners for gt_bbox_3d in gt_bboxes_3d])

        target_labels = dict(
            gt_centers_2d=batch_centers_2d.long(),
            gt_labels_3d=batch_labels_3d,
            indices=inds,
            reg_indices=reg_inds,
            gt_locs=batch_gt_locations,
            gt_dims=gt_dimensions,
            gt_yaws=gt_orientations,
            gt_cors=gt_corners)

        return center_heatmap_target, avg_factor, target_labels
