@echo off
setlocal

call activate radiomics

set INPUT="./pickle datasets"
set OUTPUT="./features"

echo "RADCURE"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type ct-clip --cohort radcure --gpu 0"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type ct-fm --cohort radcure --gpu 1"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type suprem --cohort radcure --gpu 2"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type vista3d --cohort radcure --gpu 3"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type radiomics --cohort radcure"

echo "Head Neck PET-CT"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type ct-clip --cohort headneckpetct --gpu 0"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type ct-fm --cohort headneckpetct --gpu 1"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type suprem --cohort headneckpetct --gpu 2"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type vista3d --cohort headneckpetct --gpu 3"
start cmd /k "python features.py --overwrite --input %INPUT% --output %OUTPUT% --type radiomics --cohort headneckpetct"

endlocal
