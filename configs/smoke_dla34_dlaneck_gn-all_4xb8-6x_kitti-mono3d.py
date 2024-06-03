_base_ = 'mmdet3d::smoke/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py'

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)

train_cfg = dict(val_interval=5)
auto_scale_lr = dict(enable=True, base_batch_size=32)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
