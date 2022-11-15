export CHECKPOINT=psmall33-250e
python experiment/manage_neptune_sha.py -c experiment/configs/dettrials.csv -s experiment/configs/detsha.csv \
    -e execution/smm2_det_hpo.sh -p det-${CHECKPOINT} -t det ${CHECKPOINT}