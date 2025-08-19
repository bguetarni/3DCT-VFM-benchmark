@echo off
setlocal

:: Check if argument is provided
if "%~1"=="" (
    echo Please provide an argument.
    exit /b 1
)

call activate radiomics

:: Store the argument in a variable
set "arg=%~1"

:: Compare against multiple values
if /i "%arg%"=="artix" (
    python build.py --input "C:\Users\bilel.guetarni\Desktop\data\ARTIX\DICOM_ARTIX_data" --output "C:\Users\bilel.guetarni\Desktop\ARTIX\data\artix.pkl" --cohort artix --id_map "C:\Users\bilel.guetarni\Desktop\data\ARTIX\ARTIX_ID_CORRELATION.xlsx" --clinical "C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data"
) else if /i "%arg%"=="tcia" (
    python build.py --input "C:\Users\bilel.guetarni\Desktop\data\TCIA\HNSCC-3DCT-RT\manifest-1549495779734\HNSCC-3DCT-RT" --output "C:\Users\bilel.guetarni\Desktop\ARTIX\data\tcia.pkl" --cohort tcia --clinical "C:\Users\bilel.guetarni\Desktop\data\TCIA\HNSCC-3DCT-RT\TCIA 3-6M CTCAE grade.xlsx"
) else (
    echo Unknown value: %arg%
)

endlocal
