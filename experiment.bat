@echo off
setlocal

call activate radiomics

:: set arguments
set CODE=016
set OUTPUT=C:/Users/bilel.guetarni/Desktop/workspace/SEQ-RT/experiments/%CODE%
set TASK=R2y
set BOOTSTRAP=10
set N_ITER=10000
set EVAL_ITER=100

:: linear classifier
set BISZE=64
set LR=1e-3
set "CMD=python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors ct-fm --classifier linear --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors suprem --classifier linear --normalizer scale --lr %LR% --bsize %BISZE%  --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors vista3d --classifier linear --normalizer scale --lr %LR% --bsize %BISZE%  --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb --classifier linear --normalizer scale --lr %LR% --bsize %BISZE%  --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors llm-BioLORD-2023-M --classifier linear --normalizer scale --lr %LR% --bsize %BISZE%  --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors llm-embeddinggemma-300m-medical --classifier linear --normalizer scale --lr %LR% --bsize %BISZE%  --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 0"
start cmd /k "%CMD%"

:: concat classifier
set BISZE=32
set LR=1e-3
set "CMD=python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors ct-fm,llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors ct-fm,llm-BioLORD-2023-M --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors ct-fm,llm-embeddinggemma-300m-medical --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors suprem,llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors suprem,llm-BioLORD-2023-M --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors suprem,llm-embeddinggemma-300m-medical --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors vista3d,llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors vista3d,llm-BioLORD-2023-M --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors vista3d,llm-embeddinggemma-300m-medical --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors ct-fm,suprem,vista3d,llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb,llm-BioLORD-2023-M,llm-embeddinggemma-300m-medical --classifier concat --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 1"
start cmd /k "%CMD%"

:: attention classifier
set BISZE=32
set LR=1e-3
set "CMD=python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors ct-fm,llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors ct-fm,llm-BioLORD-2023-M --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors ct-fm,llm-embeddinggemma-300m-medical --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors suprem,llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors suprem,llm-BioLORD-2023-M --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors suprem,llm-embeddinggemma-300m-medical --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors vista3d,llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors vista3d,llm-BioLORD-2023-M --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors vista3d,llm-embeddinggemma-300m-medical --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap %BOOTSTRAP% --extractors ct-fm,suprem,vista3d,llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb,llm-BioLORD-2023-M,llm-embeddinggemma-300m-medical --classifier attention --normalizer scale --lr %LR% --bsize %BISZE% --n_iter %N_ITER% --eval_iter %EVAL_ITER% --uniform_sampling --gpu 2"
start cmd /k "%CMD%"

endlocal
