WORKDIR=work_dirs/det/dettest4
python tools/benchmarks/mmdetection/testdet.py \
    --work-dir ${WORKDIR} \
    --out ${WORKDIR}/eval.pkl \
    --eval bbox proposal \
    --show --show-dir ${WORKDIR}/imgs \
    configs/det/det_example.py work_dirs/det/dettest4/latest.pth