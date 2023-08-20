cancer=BRCA
info_dir=/home/omarelnahhas/omars_hdd/omar/omar/multi-regression/TCGA-BRCA-DX_features
proj_dir=/home/omarelnahhas/omars_hdd/omar/omar/multi-regression_project
python -m barspoon.train \
    --output-dir ${proj_dir}/TME_pred_BRCA \
    --clini-table ${proj_dir}/immune_class.xlsx \
    --slide-table ${info_dir}/TCGA-${cancer}-DX_SLIDE.csv \
    --feature-dir ${info_dir}/E2E_macenko_ctranspath \
    --regr-target "Leukocyte Fraction" \
    --regr-target "LISS" \
    --regr-target "TIL Regional Fraction" \
    --regr-target "Proliferation" \
    --regr-target "Stromal Fraction" \
    --batch-size 8 \
    --max-epochs 32