checkpoint_config = dict(interval=4, max_keep_ckpts=2)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='NeptuneLoggerHook', interval=20, init_kwargs=dict(
            project='jeffjewett27/SMM2-SSOD',
            name='dettest4',
            custom_run_id='dettest4'
        )),
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
