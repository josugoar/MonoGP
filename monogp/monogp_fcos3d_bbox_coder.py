import torch
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models.task_modules import PGDBBoxCoder
from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class MonoGpFCOS3DBBoxCoder(PGDBBoxCoder):

    def decode_2d(self,
                  bbox: Tensor,
                  scale: tuple,
                  stride: int,
                  max_regress_range: int,
                  training: bool,
                  pred_shift_height: bool = False,
                  pred_keypoints: bool = False,
                  pred_bbox2d: bool = True) -> Tensor:
        clone_bbox = bbox.clone()
        if pred_shift_height:
            scale_shift_height = scale[3]
            bbox[:, self.bbox_code_size - 1] = scale_shift_height(
                clone_bbox[:, self.bbox_code_size - 1]).float()

        if pred_keypoints:
            scale_kpts = scale[3]
            if pred_shift_height:
                scale_kpts = scale[4]
            # 2 dimension of offsets x 8 corners of a 3D bbox
            bbox[:, self.bbox_code_size:self.bbox_code_size + 16] = \
                torch.tanh(scale_kpts(clone_bbox[
                    :, self.bbox_code_size:self.bbox_code_size + 16]).float())

        if pred_bbox2d:
            scale_bbox2d = scale[-1]
            # The last four dimensions are offsets to four sides of a 2D bbox
            bbox[:, -4:] = scale_bbox2d(clone_bbox[:, -4:]).float()

        if self.norm_on_bbox:
            if pred_bbox2d:
                bbox[:, -4:] = F.relu(bbox.clone()[:, -4:])
            if not training:
                if pred_keypoints:
                    bbox[
                        :, self.bbox_code_size:self.bbox_code_size + 16] *= \
                           max_regress_range
                if pred_bbox2d:
                    bbox[:, -4:] *= stride
        else:
            if pred_bbox2d:
                bbox[:, -4:] = bbox.clone()[:, -4:].exp()
        return bbox
