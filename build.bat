@echo off
setlocal
call activate radiomics
set OUTPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\pickle datasets"
python build.py --overwrite --input "E:\bilel\ARTIX\ARTIX" --output %OUTPUT% --cohort artix
python build.py --overwrite --input "E:\bilel\HECKTOR 2025 Training Data" --output %OUTPUT% --cohort hecktor
python build.py --overwrite --input "F:\TCIA\Head-Neck-CT-Atlas" --output %OUTPUT% --cohort headneckctatlas
python build.py --overwrite --input "F:\TCIA\Head-Neck-PET-CT" --output %OUTPUT% --cohort headneckpetct
python build.py --overwrite --input "F:\TCIA\HNSCC-3DCT-RT" --output %OUTPUT% --cohort hnscc3dctrt
python build.py --overwrite --input "F:\TCIA\Oropharyngeal-Radiomics-Outcomes" --output %OUTPUT% --cohort oropharyngealradiomicsoutcomes
python build.py --overwrite --input "F:\TCIA\QIN-HEADNECK" --output %OUTPUT% --cohort qinheadneck
python build.py --overwrite --input "F:\TCIA\RADCURE" --output %OUTPUT% --cohort radcure
endlocal
