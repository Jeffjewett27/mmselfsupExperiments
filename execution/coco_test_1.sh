CONFIG_FILE=configs/selfsup/densecl/densecl_resnet50_1xb32-coslr-200e_in1k.py
GPUS=1
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS}