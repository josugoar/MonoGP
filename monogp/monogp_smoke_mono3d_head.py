from typing import List, Optional, Tuple

import torch
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models import SMOKEMono3DHead
from mmdet3d.registry import MODELS
from mmdet3d.utils import InstanceList


@MODELS.register_module()
class MonoGpSMOKEMono3DHead(SMOKEMono3DHead):

    def __init__(self,
                 *args,
                 use_ground_plane: bool = False,
                 pred_shift_height: bool = False,
                 origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_ground_plane = use_ground_plane
        self.pred_shift_height = pred_shift_height
        self.origin = origin

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
        planes = torch.stack([
            cls_scores[0].new_tensor(img_meta['plane'])
            for img_meta in batch_img_metas
        ])
        trans_mats = torch.stack([
            cls_scores[0].new_tensor(img_meta['trans_mat'])
            for img_meta in batch_img_metas
        ])
        batch_bboxes, batch_scores, batch_topk_labels = self._decode_heatmap(
            cls_scores[0],
            bbox_preds[0],
            batch_img_metas,
            cam2imgs=cam2imgs,
            planes=planes,
            trans_mats=trans_mats,
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
                        planes: Tensor,
                        trans_mats: Tensor,
                        topk: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor, Tensor]:
        batch, channel = reg_pred.shape[0], reg_pred.shape[1]

        center_heatmap_pred = get_local_maximum(cls_score, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=topk)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        regression = transpose_and_gather_feat(reg_pred, batch_index)
        regression = regression.view(-1, channel)

        points = torch.cat([topk_xs.view(-1, 1),
                            topk_ys.view(-1, 1).float()],
                           dim=1)
        locations, dimensions, orientations = self.bbox_coder.decode(
            regression, points, batch_topk_labels, cam2imgs, planes,
            trans_mats, self.use_ground_plane, self.pred_shift_height,
            self.origin)

        batch_bboxes = torch.cat((locations, dimensions, orientations), dim=1)
        batch_bboxes = batch_bboxes.view(batch, -1, self.bbox_code_size)
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
        planes = torch.stack([
            gt_locations.new_tensor(img_meta['plane'])
            for img_meta in batch_img_metas
        ])
        trans_mats = torch.stack([
            gt_locations.new_tensor(img_meta['trans_mat'])
            for img_meta in batch_img_metas
        ])
        centers_2d_inds = centers_2d[:, 1] * w + centers_2d[:, 0]
        centers_2d_inds = centers_2d_inds.view(batch, -1)
        pred_regression = transpose_and_gather_feat(pred_reg, centers_2d_inds)
        pred_regression_pois = pred_regression.view(-1, channel)
        locations, dimensions, orientations = self.bbox_coder.decode(
            pred_regression_pois, centers_2d, labels_3d, cam2imgs, planes,
            trans_mats, self.use_ground_plane, self.pred_shift_height,
            self.origin, gt_locations)

        locations, dimensions, orientations = locations[indices], dimensions[
            indices], orientations[indices]

        locations[:, 1] += dimensions[:, 1] / 2

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
