# optimizer
optimizer = dict(type='SGD', lr=0.003, weight_decay=0.0005, momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
lr_config = dict(policy='CosineRestart', min_lr=0., periods=[20]*10, restart_weights=[1]*10)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)