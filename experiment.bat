@echo off
setlocal
call activate radiomics

:: set global arguments
set CODE=082
set OUTPUT=C:/Users/bilel.guetarni/Desktop/workspace/SEQ-RT/experiments/%CODE%
set KFOLD=3
set BOOTSTRAP=10
set EPOCHS=100
set BISZE=16
set LR=5e-5

set DATASET=hecktor
set TASK=rfs_2

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"

timeout /t 300 > NUL

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"

timeout /t 300 > NUL

set TASK=rfs_5

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"

timeout /t 300 > NUL

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"


timeout /t 300 > NUL


set DATASET=radcure
set TASK=rfs_2

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"

timeout /t 300 > NUL

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"

timeout /t 300 > NUL

set TASK=rfs_5

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"

timeout /t 300 > NUL

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"


timeout /t 300 > NUL


set DATASET=headneckpetct
set TASK=rfs_2

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"

timeout /t 300 > NUL

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"

timeout /t 300 > NUL

set TASK=rfs_5

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"

timeout /t 300 > NUL

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-clip --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-clip,ct-fm,vista3d,suprem --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"

endlocal
