_base_ = './fcos3d-bbox2d-kpts_r101-caffe-dcn_fpn_head-gn_4xb3-4x_kitti-mono3d.py'  # noqa: E501

# model settings
model = dict(
    bbox_head=dict(
        reg_branch=(
            (256, ),  # offset
            (),  # depth
            (256, ),  # size
            (256, ),  # rot
            (256, ),  # kpts
            (256, ),  # bbox2d
        ),
        use_ground_plane=True),
    train_cfg=dict(code_weight=[
        1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0
    ]))

backend_args = None

meta_keys = [
    'img_path', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img',
    'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    'num_pts_feats', 'pcd_trans', 'sample_idx', 'pcd_scale_factor',
    'pcd_rotation', 'pcd_rotation_angle', 'lidar_path',
    'transformation_3d_flow', 'trans_mat', 'affine_aug', 'sweep_img_metas',
    'ori_cam2img', 'cam2global', 'crop_offset', 'img_crop_offset',
    'resize_img_shape', 'lidar2cam', 'ori_lidar2img', 'num_ref_frames',
    'num_views', 'ego2global', 'axis_align_matrix', 'plane'
]
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
    dict(type='Resize3D', scale=(1242, 375), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='BBoxes3DToBBoxes'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ],
        meta_keys=meta_keys),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(type='Pack3DDetInputs', keys=['img'], meta_keys=meta_keys)
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

val_evaluator = [
    dict(
        type='KittiMetric',
        ann_file=_base_.data_root + 'kitti_infos_val.pkl',
        metric='bbox',
        backend_args=backend_args),
    dict(
        type='GroundPlaneKittiMetric',
        ann_file=_base_.data_root + 'kitti_infos_val.pkl',
        metric='bbox',
        prefix='Ground plane Kitti metric',
        backend_args=backend_args)
]
test_evaluator = val_evaluator
