import torch
from mmengine.structures import InstanceData

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import points_cam2img
from .utils import points_img2plane


@MODELS.register_module()
class MonoGpTest(Base3DDetector):

    def __init__(self, *args, origin=(0.5, 1.0, 0.5), **kwargs):
        super(MonoGpTest, self).__init__(*args, **kwargs)
        self.origin = origin

    def loss(self, batch_inputs, batch_data_samples): ...

    def predict(self, batch_inputs, batch_data_samples):
        results = []
        for batch_data_sample in batch_data_samples:
            metainfo = batch_data_sample.metainfo
            box_type_3d = metainfo['box_type_3d']
            cam2img = metainfo['cam2img']
            plane = metainfo['plane']

            eval_ann_info = batch_data_sample.eval_ann_info
            bboxes_3d = eval_ann_info['gt_bboxes_3d']
            labels_3d = bboxes_3d.tensor.new_tensor(eval_ann_info['gt_labels_3d'], dtype=torch.long)
            scores_3d = torch.ones_like(labels_3d)
            if self.origin[1] == 0.5:
                centers_2d = points_cam2img(bboxes_3d.gravity_center, cam2img)
            elif self.origin[1] == 1.0:
                centers_2d = points_cam2img(bboxes_3d.bottom_center, cam2img)
            else:
                raise ValueError(f'Unsupported origin {self.origin}')

            shift_height = plane[3] - bboxes_3d.bottom_height
            height = bboxes_3d.height
            centers_3d = points_img2plane(centers_2d,
                                          cam2img,
                                          plane,
                                          shift_height,
                                          height,
                                          self.origin)
            bboxes_3d.tensor[:, :3] = centers_3d

            result = InstanceData()
            result.bboxes_3d = box_type_3d(bboxes_3d.tensor, origin=self.origin)
            result.labels_3d = labels_3d
            result.scores_3d = scores_3d
            results.append(result)

        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results)
        return batch_data_samples

    def _forward(self, batch_inputs, batch_data_samples=None): ...

    def extract_feat(self, batch_inputs): ...
