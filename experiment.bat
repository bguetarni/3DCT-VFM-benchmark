@echo off
setlocal enabledelayedexpansion

call activate radiomics

:: set arguments
set CODE=016
set OUTPUT=C:/Users/bilel.guetarni/Desktop/workspace/SEQ-RT/experiments/%CODE%
set TASK=R2y

:: linear classifier
set "CMD=python experiments.py --task %TASK% --output %OUTPUT% --bootstrap 1 --extractors ct-fm --classifier linear --normalizer scale --n_class 1 --lr 1e-4 --bsize 32 --n_iter 1000 --eval_iter 20 --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap 1 --extractors suprem --classifier linear --normalizer scale --n_class 1 --lr 1e-4 --bsize 32 --n_iter 1000 --eval_iter 20 --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap 1 --extractors vista3d --classifier linear --normalizer scale --n_class 1 --lr 1e-4 --bsize 32 --n_iter 1000 --eval_iter 20 --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap 1 --extractors llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb --classifier linear --normalizer scale --n_class 1 --lr 1e-4 --bsize 32 --n_iter 1000 --eval_iter 20 --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap 1 --extractors llm-BioLORD-2023-M --classifier linear --normalizer scale --n_class 1 --lr 1e-4 --bsize 32 --n_iter 1000 --eval_iter 20 --gpu 0"
set "CMD=%CMD% & python experiments.py --task %TASK% --output %OUTPUT% --bootstrap 1 --extractors llm-embeddinggemma-300m-medical --classifier linear --normalizer scale --n_class 1 --lr 1e-4 --bsize 32 --n_iter 1000 --eval_iter 20 --gpu 0"
start cmd /k "%CMD%"

:: concat classifier
set "CMD="
for %%f in (ct-fm suprem vista3d) do (
    for %%g in (llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb llm-BioLORD-2023-M llm-embeddinggemma-300m-medical) do (
        set "CMD=!CMD! python experiments.py --task %TASK% --output %OUTPUT% --bootstrap 1 --extractors %%f,%%g --classifier concat --normalizer scale --n_class 1 --lr 1e-4 --bsize 32 --n_iter 1000 --eval_iter 20 --gpu 1 ^&"
    )
)
start cmd /k !CMD!

:: attention classifier
set "CMD="
for %%f in (ct-fm suprem vista3d) do (
    for %%g in (llm-BioBERT-mnli-snli-scinli-scitail-mednli-stsb llm-BioLORD-2023-M llm-embeddinggemma-300m-medical) do (
        set "CMD=!CMD! python experiments.py --task %TASK% --output %OUTPUT% --bootstrap 1 --extractors %%f,%%g --classifier attention --normalizer scale --n_class 1 --lr 1e-4 --bsize 32 --n_iter 1000 --eval_iter 20 --gpu 2 ^&"
    )
)
start cmd /k !CMD!

endlocal
