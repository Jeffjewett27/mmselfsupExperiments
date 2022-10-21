_base_ = [
    'models/frcnn_r50_c4.py',
    'datasets/coco500_dcl.py',
    'schedules/schedule_2x.py', 
    'default_runtime.py'
]

custom_imports = dict(
    imports=['tools.benchmarks.mmdetection.res_layer_extra_norm'],
    allow_failed_imports=False)