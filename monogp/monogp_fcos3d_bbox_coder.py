from typing import Optional, Tuple

from torch import Tensor

from mmdet3d.models.task_modules import FCOS3DBBoxCoder
from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class MonoGpFCOS3DBBoxCoder(FCOS3DBBoxCoder):

    def __init__(self,
                 base_depths: Optional[Tuple[Tuple[float]]] = None,
                 base_dims: Optional[Tuple[Tuple[float]]] = None,
                 code_size: int = 7,
                 norm_on_bbox: bool = True) -> None:
        super(MonoGpFCOS3DBBoxCoder, self).__init__()
        self.base_depths = base_depths
        self.base_dims = base_dims
        self.bbox_code_size = code_size
        self.norm_on_bbox = norm_on_bbox

    def decode(self,
               bbox: Tensor,
               scale: tuple,
               stride: int,
               training: bool,
               cls_score: Optional[Tensor] = None) -> Tensor:
        # scale the bbox of different level
        # only apply to offset, depth and size prediction
        scale_offset, scale_depth, scale_size = scale[0:3]

        clone_bbox = bbox.clone()
        bbox[:, :2] = scale_offset(clone_bbox[:, :2]).float()
        bbox[:, 2] = scale_depth(clone_bbox[:, 2]).float()
        bbox[:, 3:6] = scale_size(clone_bbox[:, 3:6]).float()

        if self.base_depths is None:
            bbox[:, 2] = bbox[:, 2].exp()
        elif len(self.base_depths) == 1:  # only single prior
            mean = self.base_depths[0][0]
            std = self.base_depths[0][1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std
        else:  # multi-class priors
            assert len(self.base_depths) == cls_score.shape[1], \
                'The number of multi-class depth priors should be equal to ' \
                'the number of categories.'
            indices = cls_score.max(dim=1)[1]
            depth_priors = cls_score.new_tensor(
                self.base_depths)[indices, :].permute(0, 3, 1, 2)
            mean = depth_priors[:, 0]
            std = depth_priors[:, 1]
            bbox[:, 2] = mean + bbox.clone()[:, 2] * std

        bbox[:, 3:6] = bbox[:, 3:6].exp()
        if self.base_dims is not None:
            assert len(self.base_dims) == cls_score.shape[1], \
                'The number of anchor sizes should be equal to the number ' \
                'of categories.'
            indices = cls_score.max(dim=1)[1]
            size_priors = cls_score.new_tensor(
                self.base_dims)[indices, :].permute(0, 3, 1, 2)
            bbox[:, 3:6] = size_priors * bbox.clone()[:, 3:6]

        assert self.norm_on_bbox is True, 'Setting norm_on_bbox to False '\
            'has not been thoroughly tested for FCOS3D.'
        if self.norm_on_bbox:
            if not training:
                # Note that this line is conducted only when testing
                bbox[:, :2] *= stride

        return bbox
