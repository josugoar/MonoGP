_base_ = [
    'mmdet3d::_base_/datasets/kitti-mono3d.py',
    'mmdet3d::_base_/models/fcos3d.py',
    'mmdet3d::_base_/schedules/mmdet-schedule-1x.py',
    'mmdet3d::_base_/default_runtime.py'
]

# model settings
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    bbox_head=dict(
        num_classes=3,
        bbox_code_size=7,
        pred_attrs=False,
        pred_velo=False,
        group_reg_dims=(2, 1, 3, 1),  # offset, depth, size, rot
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
        ),
        centerness_branch=(256, ),
        bbox_coder=dict(
            base_depths=((28.01, 16.32), ),
            base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6), (3.9, 1.56, 1.6)),
            code_size=7)),
    test_cfg=dict(nms_pre=100, nms_thr=0.05, score_thr=0.001, max_per_img=20))

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='mmdet.Resize', scale=(1242, 375), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=8, num_workers=4, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8, num_workers=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=48,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]

train_cfg = dict(max_epochs=48, val_interval=2)
auto_scale_lr = dict(enable=True, base_batch_size=12)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
