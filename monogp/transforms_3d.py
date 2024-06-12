import torch
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import box3d_to_bbox, points_cam2img


@TRANSFORMS.register_module()
class BBoxes3DToBBoxes(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        bboxes_3d = input_dict['gt_bboxes_3d']
        cam2img = input_dict['cam2img']

        bboxes = box3d_to_bbox(bboxes_3d.tensor.numpy(force=True), cam2img)

        input_dict['gt_bboxes'] = bboxes

        return input_dict


@TRANSFORMS.register_module()
class BottomCenterToCenters2DWithDepth(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        bboxes_3d = input_dict['gt_bboxes_3d']
        cam2img = input_dict['cam2img']

        centers_2d_with_depth = points_cam2img(
            bboxes_3d.bottom_center.numpy(force=True),
            cam2img,
            with_depth=True)

        input_dict['centers_2d'] = centers_2d_with_depth[:, :2]
        input_dict['depths'] = centers_2d_with_depth[:, 2]

        return input_dict


# TODO: FIX!!!!!!!!!!!!
@TRANSFORMS.register_module()
class ObjectShiftHeight(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        bboxes_3d = input_dict['gt_bboxes_3d']
        plane = input_dict['plane']
        box_type_3d = input_dict['box_type_3d']

        shift_height = plane[3] - bboxes_3d.bottom_height
        bboxes_3d = box_type_3d(
            torch.cat(
                [bboxes_3d.tensor, shift_height.unsqueeze(1)], dim=1),
            box_dim=bboxes_3d.box_dim + 1,
            with_yaw=bboxes_3d.with_yaw)

        input_dict['gt_bboxes_3d'] = bboxes_3d

        return input_dict
