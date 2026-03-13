@echo off
setlocal
call activate radiomics

:: set global arguments
set CODE=072
set OUTPUT=C:/Users/bilel.guetarni/Desktop/workspace/SEQ-RT/experiments/%CODE%
set KFOLD=3
set BOOTSTRAP=10
set EPOCHS=100
set BISZE=16
set LR=5e-5

set DATASET=hecktor
set TASK=rfs_2

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"

:: Cox pretraining 1v1 strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"

:: Cox pretraining 1vN strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"

set TASK=rfs_5

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 0"

:: Cox pretraining 1v1 strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 1"

:: Cox pretraining 1vN strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 2"

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler  --bsize %BISZE% --undersampling --gpu 3"


timeout /t 7200 > NUL


set DATASET=radcure
set TASK=rfs_2

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"

:: Cox pretraining 1v1 strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"

:: Cox pretraining 1vN strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"


set TASK=rfs_5

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"

:: Cox pretraining 1v1 strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"

:: Cox pretraining 1vN strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"


timeout /t 7200 > NUL


set DATASET=headneckpetct
set TASK=rfs_2

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"

:: Cox pretraining 1v1 strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"

:: Cox pretraining 1vN strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"

set TASK=rfs_5

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 0"

:: Cox pretraining 1v1 strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1v1 --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 1"

:: Cox pretraining 1vN strategy
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"
start cmd /k "python experiments.py --pretraining cox --cox_strategy 1vN --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 2"

:: ProtoNet pretraining
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality image --extractors ct-fm --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality clinical --backbone ffn --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone concat --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone attention --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"
start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --dataset %DATASET% --epoch %EPOCHS% --kfold %KFOLD% --modality both --extractors ct-fm --backbone gated --normalizer scale --lr %LR% --lr_scheduler --bsize %BISZE% --undersampling --gpu 3"

endlocal
