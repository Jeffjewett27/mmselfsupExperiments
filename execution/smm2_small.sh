CONFIG_FILE=configs/pretrain/smm2_pretrain_small.py
GPUS=1
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work-dir work_dirs/pretrain/${TRIAL}