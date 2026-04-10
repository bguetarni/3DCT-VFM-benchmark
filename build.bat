@echo off
setlocal

call activate radiomics

set OUTPUT="./pickle datasets"

echo "headneckpetct"
python build.py --overwrite --input "F:\TCIA\Head-Neck-PET-CT" --output %OUTPUT% --cohort headneckpetct

echo "radcure"
python build.py --input "F:\TCIA\RADCURE" --output %OUTPUT% --cohort radcure

endlocal
