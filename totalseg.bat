@echo off
setlocal

:: to run multiple tasks at once:
:: totalseg.bat cohort task1 & totalseg.bat cohort task2 & totalseg.bat cohort task3

:: Check if argument is provided
if "%~1" == "" (
    echo "Please provide a first argument as the cohort."
    exit /b 1
)

if "%~2" == "" (
    echo "Please provide a second argument as a valid TotalSegmentator task (head_glands_cavities, headneck_muscles, craniofacial_structures)."
    exit /b 1
)

if "%~3" == "" (
    echo "GPU id not provided, will use 0 as default"
    set gpu=0
) else (
    set "gpu=%~3"
)

call activate radiomics

:: Store the argument in a variable
set "cohort=%~1"
set "task=%~2"

:: Compare against multiple values
if "%cohort%" == "artix" (
    python totalseg.py --input "C:\Users\bilel.guetarni\Desktop\ARTIX\data\artix.pkl" --output "C:\Users\bilel.guetarni\Desktop\ARTIX\data\totalsegmentator\artix" --nii_path "C:\Users\bilel.guetarni\Desktop\ARTIX\data\nifti\artix" --task "%task%" --gpu "%gpu%"
) else if "%cohort%" == "tcia" (
    python totalseg.py --input "C:\Users\bilel.guetarni\Desktop\ARTIX\data\tcia.pkl" --output "C:\Users\bilel.guetarni\Desktop\ARTIX\data\totalsegmentator\tcia" --nii_path "C:\Users\bilel.guetarni\Desktop\ARTIX\data\nifti\tcia" --task "%task%" --gpu "%gpu%"
) else (
    echo "Unknown value: " %cohort% "use one of [artix, tcia]"
)

endlocal

call conda deactivate