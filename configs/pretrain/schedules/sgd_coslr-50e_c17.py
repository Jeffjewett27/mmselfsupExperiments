import os
import configs.environment as env

# optimizer
optimizer = dict(type='SGD', lr=env.lr, weight_decay=env.decay, momentum=env.momentum)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
lr_config = env.schedule

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=env.epochs)
