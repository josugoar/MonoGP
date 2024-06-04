from mmcv.transforms import BaseTransform

from mmdet3d.structures import box3d_to_bbox, points_cam2img
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BBoxes3DToBBoxes(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        bboxes_3d = input_dict['gt_bboxes_3d']
        cam2img = input_dict['cam2img']

        bboxes = bboxes_3d.tensor.new_tensor(
            box3d_to_bbox(bboxes_3d.tensor.numpy(force=True), cam2img))

        input_dict['gt_bboxes'] = bboxes

        return input_dict


@TRANSFORMS.register_module()
class BottomCenterToCenters2DWithDepth(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        cam2img = input_dict['cam2img']

        centers_2d_with_depth = points_cam2img(gt_bboxes_3d.bottom_center,
                                               cam2img,
                                               with_depth=True)

        input_dict['centers_2d'] = centers_2d_with_depth[:, :2]
        input_dict['depths'] = centers_2d_with_depth[:, 2]

        return input_dict


@TRANSFORMS.register_module()
class ObjectPlaneAlignment(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        cam2img = input_dict['cam2img']
        plane = input_dict['plane']

        gt_bboxes_3d.tensor[:, 4] += plane[3] - gt_bboxes_3d.bottom_height
        gt_bboxes_3d.tensor[:, 1] = plane[3]

        centers_2d_with_depth = points_cam2img(gt_bboxes_3d.gravity_center,
                                               cam2img,
                                               with_depth=True)

        input_dict['centers_2d'] = centers_2d_with_depth[:, :2]
        input_dict['depths'] = centers_2d_with_depth[:, 2]

        return input_dict
