import configs.environment as env

# model settings
model = dict(
    type='DenseCL',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    loss_lambda=env.lossLambda,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],
        norm_cfg=dict(type='BN'),
        init_cfg=env.initCfg
    ),
    neck=dict(
        type='DenseCLNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_grid=None),
    head=dict(type='ContrastiveHead', temperature=0.2))
