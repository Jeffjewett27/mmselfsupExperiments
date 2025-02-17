# model settings
model = dict(
    type='DenseCL',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    loss_lambda=0.5,
    backbone=dict(
        type='ResNet',
        depth=34,
        in_channels=3,
        out_indices=[4],
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='DenseCLNeck',
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        num_grid=None),
    head=dict(type='ContrastiveHead', temperature=0.2))
