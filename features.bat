@echo off
setlocal

:: to run multiple tasks at once:
:: XXXX.bat args1 args2 task1 & XXXX.bat args1 args2 & XXXX.bat args1 args2

:: Check if argument is provided
if "%~1" == "" (
    echo "Please provide a first argument as the cohort. (artix, tcia)"
    exit /b 1
)
if "%~2" == "" (
    echo "Please provide a second argument as a valid GPU id (0, ...)."
    exit /b 1
)

call activate radiomics

:: Store the argument in a variable
set "cohort=%~1"
set "gpu=%~2"

:: Compare against multiple values
if "%cohort%" == "artix" (
    call python features.py --input "C:\Users\bilel.guetarni\Desktop\ARTIX\data\artix.pkl" --output "C:\Users\bilel.guetarni\Desktop\ARTIX\features\original\artix" --tmp_folder "C:\Users\bilel.guetarni\Desktop\tmp\artix" --radiomics "C:\Users\bilel.guetarni\Desktop\ARTIX\radiomics.yaml" --dosiomics "C:\Users\bilel.guetarni\Desktop\ARTIX\dosiomics.yaml" --dvh --deepNN --oar_source original --oar_names "C:\Users\bilel.guetarni\Desktop\ARTIX\renamed_oars.csv" --gpu "%gpu%"
    call python features.py --input "C:\Users\bilel.guetarni\Desktop\ARTIX\data\artix.pkl" --output "C:\Users\bilel.guetarni\Desktop\ARTIX\features\totalsegmentator\artix" --tmp_folder "C:\Users\bilel.guetarni\Desktop\tmp\artix" --radiomics "C:\Users\bilel.guetarni\Desktop\ARTIX\radiomics.yaml" --dosiomics "C:\Users\bilel.guetarni\Desktop\ARTIX\dosiomics.yaml" --dvh --deepNN --oar_source totalsegmentator --gpu "%gpu%"

) else if "%cohort%" == "tcia" (
    call python features.py --input "C:\Users\bilel.guetarni\Desktop\ARTIX\data\tcia.pkl" --output "C:\Users\bilel.guetarni\Desktop\ARTIX\features\totalsegmentator\tcia" --tmp_folder "C:\Users\bilel.guetarni\Desktop\tmp\tcia" --radiomics "C:\Users\bilel.guetarni\Desktop\ARTIX\radiomics.yaml" --dosiomics "C:\Users\bilel.guetarni\Desktop\ARTIX\dosiomics.yaml" --dvh --deepNN --oar_source totalsegmentator --gpu "%gpu%"
) else (
    echo "Unknown value: " %cohort% "use one of [artix, tcia]"
)

endlocal
