_base_ = './smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py'

# model settings
model = dict(
    bbox_head=dict(
        group_reg_dims=(9, ), use_ground_plane=True, pred_shift_height=True))

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
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RandomShiftScale', shift_scale=(0.2, 0.4), aug_prob=0.3),
    dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ],
        meta_keys=meta_keys),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
    dict(type='Pack3DDetInputs', keys=['img'], meta_keys=meta_keys)
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
