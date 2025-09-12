To compute features you must **1** build the cohort object, and then **2** compute the features.

 # 1. Build a cohort
 
 Run the `build.py` script to build a pickle object containing all the DICOM data (CT, CBCT, RTDOSE, RTSTRUCT) of all patients by parsing the folder. Use the `--input` argument to provide the location of the folder and the `--output` argument to specify where to save the pickle object (i.e., list of patients).

 You can modify the way the script parse the folder by changing the `load_patient` and `load_folder` functions in the script.

Look at the `build.bat` script as an example on how to provide the arguments.

# 2. Compute features

To compute the features run the `features.py` script by providing the path to the pickle file built previously using the `--input` argument and the `--output` to specify where to save the features.

To select which features to calculate, use the following arguments:
 - `--radiomics` provide the YAML config file for calculating radiomics with  pyradiomics
 - `--dosiomics` provide the YAML config file for calculating dosiomics with  pyradiomics
 - `--dvh` just add this argument if you want to calculate DVH features (no value to be provided)
 - `--deepNN` provide the name of the foundation model to use to cumpute deep learning features (for now only 'ct-fm' is availbale)

The result is a folder containing one folder per patient with each type of features saved as a .csv file.

Look at the `features.bat` bat script as an example on how to provide the arguments.
