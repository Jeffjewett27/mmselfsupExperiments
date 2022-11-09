CONFIG_FILE=configs/pretrain/test_config.py
GPUS=1
PRETRAIN=work_dirs/pretrain/presmall3-7/extracted.pth
TRIAL='longtest2'
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} "--work-dir work_dirs/pretrain/longtest2"
# --cfg-options model.backbone.init_cfg.type=Pretrained \
# model.backbone.init_cfg.checkpoint=$PRETRAIN"
# --auto-resume \