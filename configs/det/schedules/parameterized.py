import configs.det.environment as env

# optimizer
optimizer = dict(type='SGD', lr=env.lr, momentum=env.momentum, weight_decay=env.decay)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
# learning policy
lr_config = env.lr_config

runner = dict(type='EpochBasedRunner', max_epochs=env.epochs)
