CONFIG_FILE=work_dirs/det/det-pmedium16-200e1-0/train_with_params.py
GPUS=1
PRETRAIN=models/pretrain/prelarge1-0.pth
TRIAL='det-pmedium16-200e1-0'
bash tools/benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG_FILE} ${PRETRAIN} ${GPUS} "--work-dir work_dirs/det/${TRIAL} --auto-resume"