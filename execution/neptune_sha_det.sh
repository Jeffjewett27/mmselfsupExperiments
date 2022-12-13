export CHECKPOINT=pmedium16-200e
python experiment/manage_neptune_sha.py -c experiment/configs/dettrials2.csv -s experiment/configs/detsha2.csv \
    -e execution/smm2_det_hpo.sh -p det-${CHECKPOINT} -t det ${CHECKPOINT}