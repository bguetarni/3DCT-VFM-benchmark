@echo off
setlocal

:: Check if argument is provided
if "%~1"==""(
    echo Please provide a first argument as the cohort.
    exit /b 1
)

if "%~2"==""(
    echo Please provide a second argument as a valid TotalSegmentator task (head_glands_cavities, headneck_muscles, craniofacial_structures).
    exit /b 1
)

call activate radiomics

:: Store the argument in a variable
set "cohort=%~1"
set "task=%~2"

:: Compare against multiple values
if /i "%cohort%"=="artix" (
    python totalseg.py --input "C:\Users\bilel.guetarni\Desktop\data\ARTIX\DICOM_ARTIX_data" --output "C:\Users\bilel.guetarni\Desktop\ARTIX\results\artix.pkl" --cohort artix --id_map "C:\Users\bilel.guetarni\Desktop\data\ARTIX\ARTIX_ID_CORRELATION.xlsx" --clinical "C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data"
) else if /i "%cohort%"=="tcia" (
    ::todo
) else (
    echo Unknown value: %cohort%
)

endlocal

call conda deactivate