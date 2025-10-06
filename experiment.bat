@echo off
setlocal

call activate radiomics

:: set arguments
set GPU=1
set OARSOURCE=totalsegmentator
set ARTIXBUILD="C:\Users\bilel.guetarni\Desktop\ARTIX\artix.pkl"
set HNSCCBUILD="C:\Users\bilel.guetarni\Desktop\ARTIX\hnscc.pkl"
set TMPFOLDER="C:\Users\bilel.guetarni\Desktop\tmp"
set RADIOMICSYAML="C:\Users\bilel.guetarni\Desktop\ARTIX\radiomics.yaml"
set DOSIOMICSYAML="C:\Users\bilel.guetarni\Desktop\ARTIX\dosiomics.yaml"

set ARTIXOUTPUT="C:\Users\bilel.guetarni\Desktop\ARTIX\features\artix_%OARSOURCE%_DA.csv"
set HNSCCOUTPUT="C:\Users\bilel.guetarni\Desktop\ARTIX\features\hnscc_%OARSOURCE%_DA.csv"

:: build datasets
python build.py --overwrite True --input "C:\Users\bilel.guetarni\Desktop\data\ARTIX\DICOM_ARTIX_data" --output %ARTIXBUILD% --cohort artix --id_map "C:\Users\bilel.guetarni\Desktop\data\ARTIX\ARTIX_ID_CORRELATION.xlsx" --clinical "C:\Users\bilel.guetarni\Desktop\data\ARTIX\toxicity_data"
python build.py --overwrite True --input "C:\Users\bilel.guetarni\Desktop\data\TCIA\HNSCC-3DCT-RT\manifest-1549495779734\HNSCC-3DCT-RT" --output %HNSCCOUTPUT% --cohort tcia --clinical "C:\Users\bilel.guetarni\Desktop\data\TCIA\HNSCC-3DCT-RT\TCIA 3-6M CTCAE grade.xlsx"

:: compute features using provided OAR segmentation source
if %OARSOURCE% == original (
    python features.py --input %ARTIXBUILD% --output %ARTIXOUTPUT% --tmp_folder %TMPFOLDER% --radiomics %RADIOMICSYAML% --dosiomics %DOSIOMICSYAML% --dvh --deepNN ct-fm --oar_source original --oar_names "C:\Users\bilel.guetarni\Desktop\SEQ-RT\ARTIX_OARs_NAMES.csv" --gpu %GPU%
    python features.py --input %HNSCCBUILD% --output %HNSCCOUTPUT% --tmp_folder %TMPFOLDER% --radiomics %RADIOMICSYAML% --dosiomics %DOSIOMICSYAML% --dvh --deepNN ct-fm --oar_source original --oar_names "C:\Users\bilel.guetarni\Desktop\SEQ-RT\TCIA_OARs_NAMES.csv" --gpu %GPU%
) else (
    python features.py --input %ARTIXBUILD% --output %ARTIXOUTPUT% --tmp_folder %TMPFOLDER% --radiomics %RADIOMICSYAML% --dosiomics %DOSIOMICSYAML% --dvh --deepNN ct-fm --oar_source totalsegmentator --gpu %GPU%
    python features.py --input %HNSCCBUILD% --output %HNSCCOUTPUT% --tmp_folder %TMPFOLDER% --radiomics %RADIOMICSYAML% --dosiomics %DOSIOMICSYAML% --dvh --deepNN ct-fm --oar_source totalsegmentator --gpu %GPU%
)

:: run experiment
python experiments.py --internal ??? --external ??? --output ??? --oar ??? --features ??? --feature_reduction_N ??? --normalization ??? --bootstrap 100 --kfold None

endlocal
