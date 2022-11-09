CONFIG_FILE=configs/det/det_example.py
GPUS=1
PRETRAIN=models/pretrain/r50-test-200e.pth
TRIAL='dettest3'
bash tools/benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG_FILE} ${PRETRAIN} ${GPUS} "--work-dir work_dirs/det/${TRIAL}"
# --cfg-options model.backbone.init_cfg.type=Pretrained \
# model.backbone.init_cfg.checkpoint=$PRETRAIN"
# --auto-resume \