_base_ = './fcos3d_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py'

# model settings
model = dict(
    bbox_head=dict(
        type='PGDHead',
        pred_bbox2d=True,
        group_reg_dims=(2, 1, 3, 1, 16,
                        4),  # offset, depth, size, rot, kpts, bbox2d
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            (256, ),  # kpts
            (256, )  # bbox2d
        ),
        use_depth_classifier=False,
        pred_keypoints=True,
        weight_dim=-1,
        bbox_coder=dict(type='PGDBBoxCoder')),
    train_cfg=dict(code_weight=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0
    ]))
