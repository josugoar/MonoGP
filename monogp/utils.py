from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmdet3d.utils import array_converter

DEPTH = 10


@array_converter(
    apply_to=('centers_2d', 'height', 'cam2img', 'plane', 'shift_height'))
def points_img2plane(
    centers_2d: Union[Tensor, np.ndarray],
    height: Union[Tensor, np.ndarray],
    cam2img: Union[Tensor, np.ndarray],
    plane: Union[Tensor, np.ndarray],
    shift_height: Union[Tensor, np.ndarray] = 0,
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> Union[Tensor, np.ndarray]:
    shift_height = shift_height + height * (1.0 - origin[1])

    num_points = centers_2d.shape[0]
    depths = centers_2d.new_full((num_points, 1), DEPTH)
    points = torch.cat([centers_2d[:, :2], depths], dim=1)

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    inv_cam2img = torch.inverse(cam2img).transpose(-2, -1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = (homo_xys.unsqueeze(1) @ inv_cam2img).squeeze(1)[:, :3]

    scale = torch.unsqueeze((plane[..., 3] - shift_height) / points3D[:, 1], 1)
    centers_3d = scale * points3D

    return centers_3d
