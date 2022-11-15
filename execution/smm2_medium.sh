CONFIG_FILE=configs/pretrain/smm2_pretrain_medium.py
GPUS=1
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} "--work-dir work_dirs/pretrain/${TRIAL} --auto-resume"