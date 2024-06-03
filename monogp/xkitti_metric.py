from typing import List

from mmdet3d.evaluation import KittiMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class XKittiMetric(KittiMetric):

    def convert_annos_to_kitti_annos(self, data_infos: dict) -> List[dict]:
        data_annos = super().convert_annos_to_kitti_annos(data_infos)

        if not self.format_only:
            for annos in data_annos:
                if len(annos['instances']) == 0:
                    continue

                plane = annos['plane']
                kitti_annos = annos['kitti_annos']
                location = kitti_annos['location']

                location = plane[3] / location[:, 1:2] * location

                kitti_annos['location'] = location

        return data_annos
