@echo off
setlocal

call activate radiomics

:: set arguments
set CODE=040
set OUTPUT=C:/Users/bilel.guetarni/Desktop/workspace/SEQ-RT/experiments/%CODE%
set TASK=R2y
set KFOLD=3
set TRAIN_SIZE=0.8
set EPOCHS=300
set DATASET=hecktor

:: linear classifier
set BISZE=32
set LR=1e-4
set "CMD=python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality image --extractors ct-fm --classifier linear --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality image --extractors suprem --classifier linear --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality image --extractors vista3d --classifier linear --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality clinical --classifier linear --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "%CMD%"

:: ffn classifier
set BISZE=32
set LR=1e-4
set "CMD=python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality image --extractors ct-fm --classifier ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 3"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality image --extractors suprem --classifier ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 3"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality image --extractors vista3d --classifier ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 3"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality clinical --classifier ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "%CMD%"

:: concat classifier
set BISZE=32
set LR=1e-4
set "CMD=python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors suprem --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors vista3d --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm,suprem,vista3d --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "%CMD%"

:: attention classifier
set BISZE=32
set LR=1e-4
set "CMD=python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors suprem --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors vista3d --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --train_split %TRAIN_SIZE% --modality both --extractors ct-fm,suprem,vista3d --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "%CMD%"

endlocal
