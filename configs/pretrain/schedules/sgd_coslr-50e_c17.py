# optimizer
optimizer = dict(type='SGD', lr=0.006, weight_decay=1e-4, momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
lr_config = dict(policy='CosineRestart', min_lr=0., periods=[0, 10, 20, 30, 40], restart_weights=[1]*5)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
