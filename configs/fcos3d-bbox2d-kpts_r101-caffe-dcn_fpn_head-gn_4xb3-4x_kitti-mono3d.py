_base_ = 'mmdet3d::pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py'

custom_imports = dict(imports=['projects.MonoGP.monogp'])

# model settings
model = dict(
    backbone=dict(
        frozen_stages=1,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    neck=dict(start_level=1, num_outs=5),
    bbox_head=dict(
        type='MonoGpFCOS3DMono3DHead',
        strides=(8, 16, 32, 64, 128),
        regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384), (384, 1e8)),
        use_depth_classifier=False,
        weight_dim=-1,
        bbox_coder=dict(type='MonoGpFCOS3DBBoxCoder')))

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)

train_cfg = dict(val_interval=2)
auto_scale_lr = dict(enable=True, base_batch_size=12)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
