@echo off
setlocal

:: to run multiple tasks at once:
:: XXXX.bat args1 args2 task1 & XXXX.bat args1 args2 & XXXX.bat args1 args2

call activate radiomics

:: Store the argument in a variable
set COHORT=%1
set GPU=%2

set INPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\pickle datasets"
set OUTPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\features"
set DESCRIPTION="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\clinical description\Microsoft Copilot"

@REM call python features.py --overwrite --input %INPUT% --output %OUTPUT% --type ct-fm --cohort %COHORT% --description %DESCRIPTION% --gpu %GPU%
@REM call python features.py --overwrite --input %INPUT% --output %OUTPUT% --type suprem --cohort %COHORT% --description %DESCRIPTION% --gpu %GPU%
@REM call python features.py --overwrite --input %INPUT% --output %OUTPUT% --type model-genesis --cohort %COHORT% --description %DESCRIPTION% --gpu %GPU%
call python features.py --overwrite --input %INPUT% --output %OUTPUT% --type llm --name "sentence-transformers/embeddinggemma-300m-medical" --cohort %COHORT% --description %DESCRIPTION% --gpu %GPU%
call python features.py --overwrite --input %INPUT% --output %OUTPUT% --type llm --name "FremyCompany/BioLORD-2023-M" --cohort %COHORT% --description %DESCRIPTION% --gpu %GPU%
call python features.py --overwrite --input %INPUT% --output %OUTPUT% --type llm --name "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb" --cohort %COHORT% --description %DESCRIPTION% --gpu %GPU%

endlocal
