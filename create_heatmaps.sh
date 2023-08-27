
base=/home/omarelnahhas/omars_hdd/omar/omar/
python -m barspoon.heatmaps \
    -o ${base}/multi-regression_project/TME_pred_BRCA/all_targets/retccl \
    -f ${base}/multi-regression_project \
    -m ${base}/multi-regression_project/TME_pred_BRCA/all_targets/retccl/lightning_logs/version_2/checkpoints/*.ckpt