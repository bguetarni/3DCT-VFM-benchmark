@echo off
setlocal

:: to run multiple tasks at once:
:: XXXX.bat args1 args2 task1 & XXXX.bat args1 args2 & XXXX.bat args1 args2

call activate radiomics

:: Store the argument in a variable
set COHORT=%1
set GPU=%2

set INPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\pickle datasets"
set OUTPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\featuresbis"

call python features.py --overwrite --input %INPUT% --output %OUTPUT% --type ct-fm --cohort %COHORT% --gpu %GPU%
call python features.py --overwrite --input %INPUT% --output %OUTPUT% --type suprem --cohort %COHORT% --gpu %GPU%
call python features.py --overwrite --input %INPUT% --output %OUTPUT% --type vista3d --cohort %COHORT% --gpu %GPU%

endlocal
