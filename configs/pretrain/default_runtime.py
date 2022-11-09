import configs.environment as env

# checkpoint saving
checkpoint_config = dict(interval=5, max_keep_ckpts=2)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='NeptuneLoggerHook', interval=20, init_kwargs=dict(
            project='jeffjewett27/SMM2-SSOD',
            name=env.trial,
            custom_run_id=env.trial
        )),
    ])
        # dict(type='TensorboardLoggerHook'),
# yapf:enable

# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
