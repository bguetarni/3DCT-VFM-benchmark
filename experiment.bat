@echo off
setlocal

call activate radiomics

:: set arguments
set CODE=056
set OUTPUT=C:/Users/bilel.guetarni/Desktop/workspace/SEQ-RT/experiments/%CODE%
set TASK=R2y
set KFOLD=3
set TRAIN_SIZE=0.8
set EPOCHS=100
set DATASET=hecktor


:: ffn backbone
set BISZE=32
set LR=1e-4
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality image --extractors suprem --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality image --extractors vista3d --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality clinical --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"


:: concat backbone
set BISZE=32
set LR=1e-4
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors suprem --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors vista3d --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm,suprem,vista3d --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"


:: attention backbone
set BISZE=32
set LR=1e-4
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors suprem --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors vista3d --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm,suprem,vista3d --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 3"


:: gated backbone
set BISZE=32
set LR=1e-4
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors suprem --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors vista3d --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --cox_pretraining --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm,suprem,vista3d --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"

endlocal
