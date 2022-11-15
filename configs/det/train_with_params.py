_base_ = [
    'models/frcnn_r50_c4_params.py',
    'datasets/det_example.py',
    'schedules/parameterized.py', 
    'parameterized_runtime.py'
]

custom_imports = dict(
    imports=['tools.benchmarks.mmdetection.res_layer_extra_norm'],
    allow_failed_imports=False)