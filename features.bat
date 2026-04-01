@echo off
setlocal

:: to run multiple tasks at once:
:: XXXX.bat args1 args2 task1 & XXXX.bat args1 args2 & XXXX.bat args1 args2

call activate radiomics

set INPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\pickle datasets"
set OUTPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\features"

start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type ct-fm --cohort artix --gpu 0"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type suprem --cohort artix --gpu 1"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type vista3d --cohort artix --gpu 2"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type ct-clip --cohort artix --gpu 3"

endlocal
