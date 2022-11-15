CONFIG_FILE=configs/det/det_example.py
GPUS=1
PRETRAIN=models/pretrain/psmall48-250e.pth
TRIAL='dettest7'
bash tools/benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG_FILE} ${PRETRAIN} ${GPUS} "--work-dir work_dirs/det/${TRIAL}"