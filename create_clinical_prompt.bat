@echo off
setlocal
call activate radiomics
set OUTPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\clinical description"
echo "artix"
python create_clinical_prompt.py --overwrite --input "E:\bilel\ARTIX\ARTIX" --output %OUTPUT% --cohort artix
echo "hecktor"
python create_clinical_prompt.py --overwrite --input "E:\bilel\HECKTOR 2025 Training Data" --output %OUTPUT% --cohort hecktor
echo "headneckctatlas"
python create_clinical_prompt.py --overwrite --input "F:\TCIA\Head-Neck-CT-Atlas" --output %OUTPUT% --cohort headneckctatlas
echo "headneckpetct"
python create_clinical_prompt.py --overwrite --input "F:\TCIA\Head-Neck-PET-CT" --output %OUTPUT% --cohort headneckpetct
echo "oropharyngealradiomicsoutcomes"
python create_clinical_prompt.py --overwrite --input "F:\TCIA\Oropharyngeal-Radiomics-Outcomes" --output %OUTPUT% --cohort oropharyngealradiomicsoutcomes
echo "qinheadneck"
python create_clinical_prompt.py --overwrite --input "F:\TCIA\QIN-HEADNECK" --output %OUTPUT% --cohort qinheadneck
echo "radcure"
python create_clinical_prompt.py --overwrite --input "F:\TCIA\RADCURE" --output %OUTPUT% --cohort radcure
endlocal
