@echo off
setlocal

call activate radiomics

:: set arguments
set GPU=0
set OARSOURCE=totalsegmentator
set ARTIXBUILD="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\artix.pkl"
set HNSCCBUILD="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\hnscc.pkl"
set RADIOMICSYAML="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\radiomics.yaml"
set DOSIOMICSYAML="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\dosiomics.yaml"
set TMPFOLDER="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\tmp"

set ARTIXOUTPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\features\artix_%OARSOURCE%_noDA.csv"
set HNSCCOUTPUT="C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\features\hnscc_%OARSOURCE%_noDA.csv"

:: build datasets
echo "building datasets"
python build.py --input "E:\bilel\data\ARTIX\ARTIX\DICOM_ARTIX_data" --output %ARTIXBUILD% --cohort artix --id_map "E:\bilel\data\ARTIX\ARTIX\ARTIX_ID_CORRELATION.xlsx" --clinical "E:\bilel\data\ARTIX\ARTIX\toxicity_data"
python build.py --input "E:\bilel\data\TCIA\HNSCC-3DCT-RT\manifest-1549495779734\HNSCC-3DCT-RT" --output %HNSCCBUILD% --cohort tcia --clinical "E:\bilel\data\TCIA\HNSCC-3DCT-RT\TCIA 3-6M CTCAE grade.xlsx"

:: compute features using provided OAR segmentation source
echo "computing features"
if %OARSOURCE% == original (
    python features.py --overwrite --input %ARTIXBUILD% --output %ARTIXOUTPUT% --tmp_folder %TMPFOLDER% --radiomics %RADIOMICSYAML% --dosiomics %DOSIOMICSYAML% --dvh --deepNN ct-fm --oar_source original --oar_names "C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\ARTIX_OARs_NAMES.csv" --gpu %GPU%
    python features.py --overwrite --input %HNSCCBUILD% --output %HNSCCOUTPUT% --tmp_folder %TMPFOLDER% --radiomics %RADIOMICSYAML% --dosiomics %DOSIOMICSYAML% --dvh --deepNN ct-fm --oar_source original --oar_names "C:\Users\bilel.guetarni\Desktop\workspace\SEQ-RT\TCIA_OARs_NAMES.csv" --gpu %GPU%
) else (
    python features.py --overwrite --filterDA --input %ARTIXBUILD% --output %ARTIXOUTPUT% --tmp_folder %TMPFOLDER% --radiomics %RADIOMICSYAML% --dosiomics %DOSIOMICSYAML% --dvh --deepNN ct-fm --oar_source totalsegmentator --gpu %GPU%
    python features.py --overwrite --filterDA --input %HNSCCBUILD% --output %HNSCCOUTPUT% --tmp_folder %TMPFOLDER% --radiomics %RADIOMICSYAML% --dosiomics %DOSIOMICSYAML% --dvh --deepNN ct-fm --oar_source totalsegmentator --gpu %GPU%
)

:: run experiment
echo "running experiments"
python experiments.py --internal_data %ARTIXBUILD% --external_data %HNSCCBUILD% --internal_features %ARTIXOUTPUT% --external_features %HNSCCOUTPUT% --output "./experiments" --exp_yaml "./features.yaml" --feature_reduction_N 10 --normalization minmax --bootstrap 100

endlocal
