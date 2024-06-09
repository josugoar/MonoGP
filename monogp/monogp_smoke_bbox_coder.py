from typing import Optional, Tuple

import torch
from torch import Tensor

from mmdet3d.models.task_modules import SMOKECoder
from mmdet3d.registry import TASK_UTILS
from .utils import points_img2plane


@TASK_UTILS.register_module()
class MonoGpSMOKECoder(SMOKECoder):

    def decode(self,
               reg: Tensor,
               points: Tensor,
               labels: Tensor,
               cam2imgs: Tensor,
               trans_mats: Tensor,
               planes: Optional[Tensor] = None,
               use_ground_plane: bool = False,
               pred_shift_height: bool = False,
               origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
               locations: Optional[Tensor] = None) -> Tuple[Tensor]:
        depth_offsets = reg[:, 0]
        centers2d_offsets = reg[:, 1:3]
        dimensions_offsets = reg[:, 3:6]
        orientations = reg[:, 6:8]
        shift_heights = 0
        if pred_shift_height:
            shift_heights = reg[:, 8]
        depths = self._decode_depth(depth_offsets)
        pred_dimensions = self._decode_dimension(labels, dimensions_offsets)
        # get the 3D Bounding box's center location.
        pred_locations = self._decode_location(points, centers2d_offsets,
                                               pred_dimensions, depths,
                                               shift_heights, cam2imgs,
                                               trans_mats, planes,
                                               use_ground_plane,
                                               pred_shift_height, origin)
        if locations is None:
            pred_orientations = self._decode_orientation(
                orientations, pred_locations)
        else:
            pred_orientations = self._decode_orientation(
                orientations, locations)

        return pred_locations, pred_dimensions, pred_orientations

    def _decode_location(self, points: Tensor, centers2d_offsets: Tensor,
                         dimensions: Tensor, depths: Tensor,
                         shift_heights: Tensor, cam2imgs: Tensor,
                         trans_mats: Tensor, planes: Optional[Tensor],
                         use_ground_plane: bool, pred_shift_height: bool,
                         origin: Tuple[float, float, float]) -> Tensor:
        # number of points
        N = centers2d_offsets.shape[0]
        # batch_size
        N_batch = cam2imgs.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()
        trans_mats = trans_mats[obj_id]
        cam2imgs = cam2imgs[obj_id]
        centers2d = points + centers2d_offsets
        if use_ground_plane:
            locations = points_img2plane(
                centers2d,
                dimensions[:, 1],
                cam2imgs,
                planes,
                shift_heights,
                origin=origin)
        else:
            trans_mats_inv = trans_mats.inverse
            cam2imgs_inv = cam2imgs.inverse()
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
