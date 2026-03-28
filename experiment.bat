@echo off
setlocal
call activate radiomics

:: set global arguments
set CODE=087
set OUTPUT=C:/Users/bilel.guetarni/Desktop/workspace/SEQ-RT/experiments/%CODE%
set BOOTSTRAP=1
set EPOCHS=100
set BISZE=16
set LR=5e-5

set INTERNAL=radcure
set EXTERNAL=artix,headneckpetct,hecktor,radcure
set CLINICAL=age,sex,treatment,dose

set TASK=rfs_2

:: no pretraining
start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality clinical --backbone ffn --clinical %CLINICAL% --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
@REM start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality image --extractors ct-clip --clinical %CLINICAL% --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
@REM start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
@REM start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
@REM start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"

@REM :: ProtoNet pretraining
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality clinical --backbone ffn --clinical %CLINICAL% --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality image --extractors ct-clip --clinical %CLINICAL% --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"

@REM :: Co + ProtoNet pretraining
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality clinical --backbone ffn --clinical %CLINICAL% --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality image --extractors ct-clip --clinical %CLINICAL% --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"

@REM timeout /t 300 > NUL

@REM set TASK=rfs_5

@REM :: no pretraining
@REM start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality clinical --backbone ffn --clinical %CLINICAL% --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
@REM start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality image --extractors ct-clip --clinical %CLINICAL% --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
@REM start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
@REM start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"
@REM start cmd /k "python experiments.py --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 0"

@REM :: ProtoNet pretraining
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality clinical --backbone ffn --clinical %CLINICAL% --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality image --extractors ct-clip --clinical %CLINICAL% --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"
@REM start cmd /k "python experiments.py --pretraining protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 1"

@REM :: Cox + ProtoNet pretraining
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality clinical --backbone ffn --clinical %CLINICAL% --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality image --extractors ct-clip --clinical %CLINICAL% --backbone ffn --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone concat --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone attention --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"
@REM start cmd /k "python experiments.py --pretraining cox+protonet --task %TASK% --output %OUTPUT% --internal %INTERNAL% --external %EXTERNAL% --epoch %EPOCHS% --bootstrap %BOOTSTRAP% --modality both --extractors ct-clip --clinical %CLINICAL% --backbone gated --normalizer scale --lr %LR% --bsize %BISZE% --undersampling --gpu 2"

endlocal