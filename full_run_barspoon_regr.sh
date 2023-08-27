cancer=BRCA
info_dir=/home/omarelnahhas/omars_hdd/omar/omar/multi-regression/TCGA-BRCA-DX_features
proj_dir=/home/omarelnahhas/omars_hdd/omar/omar/multi-regression_project
python -m barspoon.train \
    --output-dir ${proj_dir}/TME_pred_BRCA/all_targets/retccl/bigahh \
    --clini-table ${proj_dir}/immune_class.xlsx \
    --slide-table /home/omarelnahhas/omars_hdd/omar/omar/immunoproject/new/TCGA-BRCA-DX-features/TCGA-BRCA-DX_SLIDE.csv \
    --feature-dir /home/omarelnahhas/omars_hdd/omar/omar/immunoproject/new/TCGA-BRCA-DX-features/e2e-xiyue-wang-macenko \
    --regr-target "Leukocyte Fraction" \
    --regr-target "Stromal Fraction" \
    --regr-target "LISS" \
    --regr-target "TIL Regional Fraction" \
    --regr-target "Proliferation" \
    --batch-size 1 \
    --patience 50 \
    --max-epochs 500 \
    --d-model 2048 \
    --dim-feedforward 2048 \
    --num-encoder-heads 16 \
    --num-encoder-layers 2 \
    --num-decoder-heads 16 \
    --num-decoder-layers 2 \
    --instances-per-bag 4096 \
    --no-positional-encoding
