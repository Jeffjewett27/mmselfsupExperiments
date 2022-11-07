CONFIG_FILE=configs/det/det_example.py
GPUS=1
PRETRAIN=work_dirs/pretrain/longtest1/extracted.pth
TRIAL='dettest1'
bash tools/benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG_FILE} ${PRETRAIN} ${GPUS} "--work-dir work_dirs/pretrain/dettest1"
# --cfg-options model.backbone.init_cfg.type=Pretrained \
# model.backbone.init_cfg.checkpoint=$PRETRAIN"
# --auto-resume \