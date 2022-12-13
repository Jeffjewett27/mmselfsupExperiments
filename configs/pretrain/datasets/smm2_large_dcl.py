# dataset settings
data_source = 'ImageList'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# The difference between mocov2 and mocov1 is the transforms in the pipeline
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.6),
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
            data_prefix='data/smm2/youtube_large/train',
            ann_file='data/smm2/youtube_large/trainList.txt',
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=False,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/smm2/youtube_large/validation',
            ann_file='data/smm2/youtube_large/valList.txt',
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=False,
        byAxis=1
    ))