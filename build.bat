@echo off
setlocal
call activate radiomics
set OUTPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\pickle datasets"

echo "artix"
python build.py --input "E:\bilel\ARTIX\ARTIX" --output %OUTPUT% --cohort artix

echo "hecktor"
python build.py --input "E:\bilel\HECKTOR 2025 Training Data" --output %OUTPUT% --cohort hecktor

echo "headneckctatlas"
python build.py --input "F:\TCIA\Head-Neck-CT-Atlas" --output %OUTPUT% --cohort headneckctatlas

echo "headneckpetct"
python build.py --overwrite --input "F:\TCIA\Head-Neck-PET-CT" --output %OUTPUT% --cohort headneckpetct

echo "hnscc3dctrt"
python build.py --input "F:\TCIA\HNSCC-3DCT-RT" --output %OUTPUT% --cohort hnscc3dctrt

echo "oropharyngealradiomicsoutcomes"
python build.py --input "F:\TCIA\Oropharyngeal-Radiomics-Outcomes" --output %OUTPUT% --cohort oropharyngealradiomicsoutcomes

echo "qinheadneck"
python build.py --input "F:\TCIA\QIN-HEADNECK" --output %OUTPUT% --cohort qinheadneck

echo "radcure"
python build.py --input "F:\TCIA\RADCURE" --output %OUTPUT% --cohort radcure

endlocal
