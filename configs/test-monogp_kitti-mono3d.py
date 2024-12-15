_base_ = [
    'mmdet3d::_base_/datasets/kitti-mono3d.py',
    'mmdet3d::_base_/schedules/mmdet-schedule-1x.py',
    'mmdet3d::_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.monogp.monogp'])

model = dict(type='MonoGpTest')

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
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ],
        meta_keys=meta_keys)
]

test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

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

default_hooks = dict(visualization=dict(type='BEVDet3DVisualizationHook'))
