CONFIG_FILE=configs/det/train_with_params.py
GPUS=1
PRETRAIN=models/pretrain/${CHECKPOINT}.pth
export TRIAL=${TRIAL}
bash tools/benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG_FILE} ${PRETRAIN} ${GPUS} "--work-dir work_dirs/det/${TRIAL}"