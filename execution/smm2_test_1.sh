CONFIG_FILE=configs/pretrain/smm2_r34_1xb32-coslr-50e_c17.py
GPUS=1
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work-dir work_dirs/pretrain/${TRIAL}