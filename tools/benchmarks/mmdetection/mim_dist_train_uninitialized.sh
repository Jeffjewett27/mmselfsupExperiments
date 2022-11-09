#!/usr/bin/env bash

set -x

CFG=$1
WORK_DIR=$2  # pretrained model
GPUS=$3
PY_ARGS=${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mim train mmdet $CFG \
    --launcher pytorch -G $GPUS \
    --work-dir $WORK_DIR \
    $PY_ARGS
