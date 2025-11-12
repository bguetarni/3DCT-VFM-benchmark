@echo off
setlocal

:: to run multiple tasks at once:
:: XXXX.bat args1 args2 task1 & XXXX.bat args1 args2 & XXXX.bat args1 args2

@REM :: Check if argument is provided
@REM if "%~1" == "" (
@REM     echo "Please provide a first argument as the cohort. (artix, tcia)"
@REM     exit /b 1
@REM )
@REM if "%~2" == "" (
@REM     echo "Please provide a second argument as a valid GPU id (0, ...)."
@REM     exit /b 1
@REM )

call activate radiomics

@REM :: Store the argument in a variable
@REM set "cohort=%~1"
@REM set "gpu=%~2"

set OUTPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\features"
set TMP="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\tmp"

call python features.py --overwrite --input "C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\pickle datasets\headneckctatlas.pickle" --output %OUTPUT% --tmp_folder %TMP% --type model-genesis --cohort headneckctatlas --gpu 0

endlocal
