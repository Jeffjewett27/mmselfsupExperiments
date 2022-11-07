_base_ = [
    'models/frcnn_r50_c4.py',
    'datasets/det_example.py',
    'schedules/det_example.py', 
    'default_runtime.py'
]

custom_imports = dict(
    imports=['tools.benchmarks.mmdetection.res_layer_extra_norm'],
    allow_failed_imports=False)