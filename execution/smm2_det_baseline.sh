CONFIG_FILE=configs/det/train_with_params.py
GPUS=1
WORK_DIR="work_dirs/det/${TRIAL}"
bash tools/benchmarks/mmdetection/mim_dist_train_uninitialized.sh ${CONFIG_FILE} ${WORK_DIR} ${GPUS} "--work-dir ${WORK_DIR}"
# --cfg-options model.backbone.init_cfg.type=Pretrained \
# model.backbone.init_cfg.checkpoint=$PRETRAIN"
# --auto-resume \