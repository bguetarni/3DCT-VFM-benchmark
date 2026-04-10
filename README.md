# GitHub repository for 3D CT Vision Foundation Models benchmarking on H&N cancer

To run the benchmark you need to:
 - calculate the fatures of each datasets (provided in the `features` folder)
    - download the datasets in the TCIA platform ([RADCURE](https://www.cancerimagingarchive.net/collection/radcure/) and [Head-Neck-PET-CT](https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/))
    - run `build.bat` to build the datasets pickle file
    - run `features.bat` to compute the features
    - run `experiment.bat` for running training and testing

Models training, validation and test metrics will be saved under `experiments/{model_name}/metrics.csv`.

> **Note**: TCIA has interrupted access to data, so we provide access to the features pre-computed !
