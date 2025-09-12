@echo off
setlocal
call activate radiomics
call python features.py --input "path/to/cohort.pkl" --output "path/to/output/folder" --radiomics "path/to/radiomics.yaml" --dosiomics "path/to/dosiomics.yaml" --dvh --deepNN ct-fm --gpu 0
endlocal
