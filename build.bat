@echo off
setlocal
call activate radiomics
python build.py --input "path/to/folder/dicom" --output "./artix.pkl"
endlocal
