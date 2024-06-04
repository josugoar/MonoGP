from typing import Tuple, Union

import torch
import numpy as np
from torch import Tensor

from mmdet3d.structures import points_img2cam
from mmdet3d.utils import array_converter


@array_converter(apply_to=('centers_2d', 'cam2img', 'plane'))
def points_img2plane(centers_2d, cam2img, plane, shift_height, height, origin):
    # Arbitrary depth value representing the scale to a 3D plane in the camera
    # coordinate system. Careful with its magnitude, as a low value may cause
    # numerical instability
    DEPTH = 100

    shift_height += height * (1.0 - origin[1])

    num_points = centers_2d.shape[0]
    depths = centers_2d.new_full((num_points, 1), DEPTH)
    points = torch.cat([centers_2d, depths], dim=1)
    points3D = points_img2cam(points, cam2img)
    scale = torch.unsqueeze((plane[3] - shift_height) / points3D[:, 1], 1)
    centers_3d = scale * points3D

    return centers_3d
