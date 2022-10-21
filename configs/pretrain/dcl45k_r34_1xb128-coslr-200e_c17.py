_base_ = [
    'models/dcl_r34.py',
    'datasets/coco_dcl_45k.py',
    'schedules/sgd_coslr-50e_c17.py',
    'default_runtime.py',
]
