import configs.environment as env

# model settings
model = dict(
    type='DenseCL',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    loss_lambda=0.5,
    backbone=dict(
        type='ResNet',
        depth=env.depth,
        in_channels=3,
        out_indices=[4],
        norm_cfg=dict(type='BN')),
        init_cfg=dict(
            type=env.init, distribution='uniform'),
    neck=dict(
        type='DenseCLNeck',
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        num_grid=None),
    head=dict(type='ContrastiveHead', temperature=0.2))
