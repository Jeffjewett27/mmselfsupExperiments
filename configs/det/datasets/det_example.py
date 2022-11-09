# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/smm2/youtube_labeled/'
img_norm_cfg = dict(
    mean=[100.4, 108.63, 102.6], std=[62.86, 57.78, 59.69], to_rgb=True)
classes=('mario','luigi','toad','toadette','general_enemy','goomba','shellish','biter','thwomp','boo','boss','coin',
    'powerup','blaster','icicle','bullet','door','spring','pow','on_off_switch','moving_platform','hazard','gizmo_item')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainLabels.json',
        img_prefix=data_root + 'train/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valLabels.json',
        img_prefix=data_root + 'validation/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'testLabels.json',
        img_prefix=data_root + 'test/',
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=2, metric='bbox', save_best='auto')