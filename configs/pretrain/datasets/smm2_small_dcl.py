# dataset settings
data_source = 'ImageList'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.3937, 0.4260, 0.4053], std=[0.2465, 0.2266, 0.2341])
# The difference between mocov2 and mocov1 is the transforms in the pipeline
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.3),
    dict(type='RandomVerticalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]

val_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]

# prefetch
# prefetch = False
# if not prefetch:
#     train_pipeline.extend(
#         [dict(type='ToTensor'),
#          dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=32,  # total 32*8=256
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/smm2/youtube_small/train',
            ann_file='data/trainListSmall.txt',
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=False,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/smm2/youtube_small/validation',
            ann_file='data/valListSmall.txt',
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=False,
        byAxis=1
    ))