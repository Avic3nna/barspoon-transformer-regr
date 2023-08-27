cancer=BRCA
info_dir=/home/omarelnahhas/omars_hdd/omar/omar/multi-regression/TCGA-BRCA-DX_features
proj_dir=/home/omarelnahhas/omars_hdd/omar/omar/multi-regression_project
python -m barspoon.train \
    --output-dir ${proj_dir}/TME_pred_BRCA/path_targets_1bs \
    --clini-table ${proj_dir}/immune_class.xlsx \
    --slide-table ${info_dir}/TCGA-${cancer}-DX_SLIDE.csv \
    --feature-dir ${info_dir}/E2E_macenko_ctranspath \
    --regr-target "Leukocyte Fraction" \
    --regr-target "Stromal Fraction" \
    --regr-target "TIL Regional Fraction" \
    --batch-size 1 \
    --patience 15 \
    --max-epochs 32 \
    --d-model 768 \
    --dim-feedforward 768 \
    --num-encoder-heads 16 \
    --num-encoder-layers 4 \
    --num-decoder-heads 16 \
    --num-decoder-layers 4 \
    --instances-per-bag 4096 \
    --no-positional-encoding
