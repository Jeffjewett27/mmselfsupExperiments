WORKDIR=work_dirs/det/det-v2-prelarge1-01-0
python tools/benchmarks/mmdetection/testdet.py \
    --work-dir ${WORKDIR} \
    --out ${WORKDIR}/eval.pkl \
    --eval bbox proposal \
    --show --show-dir ${WORKDIR}/imgs \
    ${WORKDIR}/train_with_params.py ${WORKDIR}/latest.pth