from typing import List, Tuple

import numpy as np

from mmdet3d.evaluation import KittiMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import points_cam2img
from .utils import points_img2plane


@METRICS.register_module()
class GroundPlaneKittiMetric(KittiMetric):

    def __init__(self,
                 *args,
                 origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.origin = origin

    def convert_annos_to_kitti_annos(self, data_infos: dict) -> List[dict]:
        data_annos = super().convert_annos_to_kitti_annos(data_infos)

        if not self.format_only:
            for annos in data_annos:
                if len(annos['instances']) == 0:
                    continue

                cam2img = annos['images'][self.default_cam_key]['cam2img']
                plane = annos['plane']
                kitti_annos = annos['kitti_annos']
                dimensions = kitti_annos['dimensions']
                location = kitti_annos['location']

                dst = np.array((0.5, 1.0, 0.5))
                src = np.array(self.origin)

                location -= dimensions * (dst - src)
                location = points_img2plane(
                    points_cam2img(location, cam2img),
                    dimensions[:, 1],
                    cam2img,
                    plane,
                    origin=self.origin)
                location += dimensions * (dst - src)

                kitti_annos['location'] = location

        return data_annos
