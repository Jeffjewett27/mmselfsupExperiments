CONFIG_FILE=configs/det/det_example.py
GPUS=1
WORK_DIR="work_dirs/det/dettest11"
bash tools/benchmarks/mmdetection/mim_dist_train_uninitialized.sh ${CONFIG_FILE} ${WORK_DIR} ${GPUS} "--work-dir ${WORK_DIR}"
# --cfg-options model.backbone.init_cfg.type=Pretrained \
# model.backbone.init_cfg.checkpoint=$PRETRAIN"
# --auto-resume \