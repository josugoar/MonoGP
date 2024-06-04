from typing import Optional, Tuple

import torch
from torch import Tensor

from mmdet3d.models.task_modules import SMOKECoder
from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class MonoGpSMOKECoder(SMOKECoder):

    def __init__(self, base_shift_height: Tuple[float],
                 base_depth: Tuple[float], base_dims: Tuple[float],
                 code_size: int):
        super(MonoGpSMOKECoder, self).__init__(base_depth, base_dims,
                                               code_size)
        self.base_shift_height = base_shift_height

    def decode(self,
               reg: Tensor,
               points: Tensor,
               labels: Tensor,
               planes: Tensor,
               cam2imgs: Tensor,
               trans_mats: Tensor,
               pred_shift_height: bool,
               locations: Optional[Tensor] = None) -> Tuple[Tensor]:
        # TODO
        shift_height_offsets = reg[:, 0]
        centers2d_offsets = reg[:, 1:3]
        dimensions_offsets = reg[:, 3:6]
        orientations = reg[:, 6:8]
        if pred_shift_height:
            shift_heights = self._decode_shift_height(shift_height_offsets)
        else:
            shift_heights = torch.zeros_like(shift_height_offsets)
        depths = self._decode_depth(...)
        # get the 3D Bounding box's center location.
        pred_locations = self._decode_location(points, centers2d_offsets,
                                               depths, cam2imgs, trans_mats)
        pred_dimensions = self._decode_dimension(labels, dimensions_offsets)
        if locations is None:
            pred_orientations = self._decode_orientation(
                orientations, pred_locations)
        else:
            pred_orientations = self._decode_orientation(
                orientations, locations)

        return pred_locations, pred_dimensions, pred_orientations

    def _decode_shift_height(self, shift_height_offsets: Tensor) -> Tensor:
        base_shift_height = shift_height_offsets.new_tensor(
            self.base_shift_height)
        depths = shift_height_offsets * base_shift_height[
            1] + base_shift_height[0]

        return depths

    def _decode_depth(self, depth_offsets: Tensor) -> Tensor:
        base_depth = depth_offsets.new_tensor(self.base_depth)
        depths = depth_offsets * base_depth[1] + base_depth[0]

        return depths

    def _decode_location(self, points: Tensor, centers2d_offsets: Tensor,
                         depths: Tensor, cam2imgs: Tensor,
                         trans_mats: Tensor) -> Tensor:
        # number of points
        N = centers2d_offsets.shape[0]
        # batch_size
        N_batch = cam2imgs.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()
        trans_mats_inv = trans_mats.inverse()[obj_id]
        cam2imgs_inv = cam2imgs.inverse()[obj_id]
        centers2d = points + centers2d_offsets
        centers2d_extend = torch.cat((centers2d, centers2d.new_ones(N, 1)),
                                     dim=1)
        # expand project points as [N, 3, 1]
        centers2d_extend = centers2d_extend.unsqueeze(-1)
        # transform project points back on original image
        centers2d_img = torch.matmul(trans_mats_inv, centers2d_extend)
        centers2d_img = centers2d_img * depths.view(N, -1, 1)
        if cam2imgs.shape[1] == 4:
            centers2d_img = torch.cat(
                (centers2d_img, centers2d.new_ones(N, 1, 1)), dim=1)
        locations = torch.matmul(cam2imgs_inv, centers2d_img).squeeze(2)

        return locations[:, :3]
