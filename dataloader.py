from abc import ABC, abstractmethod
import glob
import os
import tqdm
import pathlib
import re
import pickle
import pydicom
import pandas
import datetime
import numpy as np
import SimpleITK as sitk
import scipy
from monai.transforms import Flip, SpatialCrop

import dicom_class
import dicom_utils

recurrence_2y_name = "R2y"
recurrence_5y_name = "R5y"

CATEGORICAL_CLINICAL_VARIABLES = ["sex", "ecog", "smoking", "stage", "hpv", "treatment"]

def convert2date(dt):
    return datetime.datetime.strptime(dt, "%d/%m/%Y").date()


class BaseLoader(ABC):
    def __init__(self, path):
        self.path = path

    def load(self, logger):
        data = self.build_patients(logger)
        data = {p.id: p for p in data}
        return data
    
    def find_digit(self, x):
        if isinstance(x, str):
            digits = re.findall("\d", x)
            if digits:
                return int(digits[0])
            else:
                return None
        else:
            return None
        
    def parse_dicom_data(self, path):
        """
        Load a folder which could be anything (collection of imaging, CT, RTDOSE, ...)

        return list of objects of type (CT, CBCT, RTDOSE, RTSTRUCT)
        """

        data = []
        try:
            if all([pydicom.misc.is_dicom(j) for j in glob.glob(os.path.join(path, "*"))]):
                # it is DICOM folder

                files = os.listdir(path)
                
                if len(files) == 0:
                    return []
                
                dcm = pydicom.dcmread(os.path.join(path, files[0]))
                type = dcm.get((0x0008, 0x0060)).value
                if type == "CT":
                    if dicom_utils.is_CT(dcm, use_exposure_time=False):
                        return [dicom_class.CT(path)]
                    else:
                        return [dicom_class.CBCT(path)]
                elif type == "RTSTRUCT":
                    return [dicom_class.RTSTRUCT(path)]
                elif type == 'RTDOSE':
                    return [dicom_class.RTDOSE(path)]
                else:
                    return []
        except PermissionError:
            pass

        for folder in glob.glob(os.path.join(path, "*")):
            if pathlib.Path(folder).is_dir():
                data.extend(self.parse_dicom_data(folder))

        return data


class ARTIX(BaseLoader):
    # CT vs CBCT: (0008, 0070) Manufacturer of CBCT is 'ELEKTA'
    def __init__(self, path):
        super().__init__(path)
        self.CLINICAL_MAPPING = {
            # SEX
            "Male": 1,
            "Female": 2,

            # CANCER STAGING
            "Stage III": 3,
            "Stage IVa": 4,
            "Stage IVb": 4,

            # ECOG
            "Asympatomatic": 0,
            "Completely ambulatory": 1,
            "lower than 50% in bed": 2,
        }

    def build_patients(self, log=None):
        data = []
        list_of_patients =  glob.glob(os.path.join(self.path, "DICOM_ARTIX_data", "*"))
        for p in tqdm.tqdm(list_of_patients):
            # convert patient ID from folder to clinical data
            id = pathlib.Path(p).name
            id_map = pandas.read_excel(os.path.join(self.path, "ARTIX_ID_CORRELATION.xlsx"))
            id = str(id_map[id_map["My Identifier ID"].astype(float) == float(id)]["USUBJID"].item()).zfill(3)
            
            # load every DICOM data of patient folder
            patient_data = self.parse_dicom_data(p)

            # group CT with dose
            for rtdose in filter(lambda i: isinstance(i, dicom_class.RTDOSE), patient_data):
                done = False
                for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
                    if max(dicom_utils.get_directory_level(rtdose.path, ct.path)) > 1:
                        continue

                    if rtdose.get_FrameOfReferenceUID() == ct.get_FrameOfReferenceUID() or \
                        rtdose.get_StudyInstanceUID() == ct.get_StudyInstanceUID() or \
                            ct.rtdose is None:
                        ct.add_rtdose(rtdose, log)
                        done = True
                        break

                if not(done) and log: log.warning(f"WARNING: RTDOSE at {rtdose.path} found no matching CT")

            # group CT with struct
            for rtstruct in filter(lambda i: isinstance(i, dicom_class.RTSTRUCT), patient_data):
                done = False
                for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
                    if (rtstruct.get_StudyID() == ct.get_StudyID() and ct.rtstruct is None) or \
                        rtstruct.get_StudyInstanceUID() == ct.get_StudyInstanceUID() or \
                            (max(dicom_utils.get_directory_level(rtstruct.path, ct.path)) < 2 and ct.rtstruct is None):
                        ct.add_rtstruct(rtstruct, log)
                        done = True
                        break

                if not(done) and log: log.warning(f"WARNING: RTSTRUCT at {rtstruct.path} found no matching CT")
            
            # parse clinical data
            clinical = {}
            for i in ("20241021_PATIENT_DESCRIPTION_LTSI.csv", "20241021_EFFICACY_LTSI.csv", "20241021_TREATMENT_LTSI.csv"):
                df = pandas.read_csv(os.path.join(self.path, "toxicity_data", i), sep=";", encoding='ISO-8859-1')
                df = df[df["USUBJID"].astype(int) == int(id)].to_dict(orient="records")
                if df:
                    clinical.update(df[0])
                else:
                    if log: log.warning(f"WARNING: patient {id} not found in clinical data {i}")

            # parse toxicity data
            tox_df = pandas.read_csv(os.path.join(self.path, "toxicity_data", "20241021_TOX_GRADE_LTSI.csv"), sep=";", encoding='ISO-8859-1')
            tox_df = tox_df[(tox_df["USUBJID"].astype(int) == int(id)) & (tox_df["AETERM"] == "XEROSTOMIE")].to_dict(orient="records")
            if tox_df:
                tox_df = tox_df[0]
                clinical.update({"baseline": tox_df["grade_S0"], "xerostomia": tox_df["grade_M6"]})
            else:
                if log: log.warning(f"WARNING: patient {id} toxicity data not found")

            # build patient object
            p = dicom_class.Patient(
                patient_id = str(id).zfill(3),
                ct = list(filter(lambda i: isinstance(i, dicom_class.CT), patient_data)),
                cbct = list(filter(lambda i: isinstance(i, dicom_class.CBCT), patient_data)),
                clinical = clinical)
            
            data.append(p)
        
        return data
    
    def generate_clinical_df(self):
        col = {
            "USUBJID":"Unique subject identifier",
            "SEX":"Gender",
            "AGE":"Age at inclusion",
            "BMI":"Body Mass Index (kg/m²)",
            "ECOG":"OMS performance status",
            "FUM":"Tabagism",
            "PA":"Number of pack-year",
            "OH":"Ethylisme",
            "DIAB":"Diabetes Mellitus",
            "P16":"p16 gene expression",
            "ST_STAG":"Strat Disease Stage",
            "ST_HPV":"Strat HPV Status",
            "HISTO":"Tumor histology",
            "LOC":"Primary tumor localization",
            "NSTAG":"N stage",
            "RT_NB":"Number of RT sessions",
            "PTV70_DOSE":"PTV70 Dose per fraction"}

        df = None
        for f in ["20241021_PATIENT_DESCRIPTION_LTSI.csv", "20241021_DOSIMETRY_LTSI.csv", "20241021_TREATMENT_LTSI.csv"]:
            f = pandas.read_csv(os.path.join(self.path, "toxicity_data", f), sep=";", encoding='ISO-8859-1')
            if "DOSISEQ" in f.columns:
                f = f[f["DOSISEQ"] == "Initial"]
            f = f[[c for c in f.columns if c in col.keys()]]
            if df is None:
                df = f
            else:
                df = df.merge(f, how='outer')

        df = df.rename(columns=col, errors="raise")
        return df
    
    def build_dataset(self, path_=None):
        """
        Return features and labels as pandas.Dataframe
        """
        base_path = pathlib.Path("..") if path_ is None else pathlib.Path(path_)

        with open(base_path.joinpath("pickle datasets", "artix.pickle"), "rb") as f:
            patients = pickle.load(f)

        # concatenate features and modalities
        features = []
        for file_ in base_path.joinpath("features", "artix").iterdir():
            fts = pandas.read_csv(file_)
            fts["modality"] = "clinical" if file_.name.startswith("llm") else "image"
            fts["features"] = file_.stem
            fts["patient"] = fts["patient"].apply(lambda i: str(i).zfill(3))
            features.append(fts)        
        features = pandas.concat(features)

        # build labels
        label = []
        for id_, p in patients.items():
            r2y, r5y = None, None

            progdt = p.clinical["PROGDT"]   # date of progression
            rtend = p.clinical["RTENDT"]   # date of end RT
            lastfollow = p.clinical["DDNDT"]   # date of last follow-up
            prog = p.clinical["PROG"]   # binary indicator of progression occuring during follow-up
            
            if pandas.isna(progdt) or pandas.isna(rtend):
                prog_endrt_days = None
            else:
                prog_endrt_days = (convert2date(progdt) - convert2date(rtend)).days

            if lastfollow is None or pandas.isna(rtend):
                endfollow_endrt_days = None
            else:
                endfollow_endrt_days = (convert2date(lastfollow) - convert2date(rtend)).days

            if prog == "Yes" and not(prog_endrt_days is None):
                r2y = int(prog_endrt_days > 2*365)
                r5y = int(prog_endrt_days > 5*365)
            elif not(endfollow_endrt_days is None):
                r2y = 1 if endfollow_endrt_days > 2*365 else None
                r5y = 1 if endfollow_endrt_days > 5*365 else None
            else:
                pass
            
            label.append({"patient": id_, 
                          "center": "CEM", # Centre Eugene Marquis
                          recurrence_2y_name: r2y,
                          recurrence_5y_name: r5y})
        label = pandas.DataFrame(label)

        return features, label

    def get_spatial_transforms():
        tr = [SpatialCrop(roi_start=(0,0,0), roi_end=(1000,1000,400)),]
        return tr


class HECKTOR(BaseLoader):
    def __init__(self, path):
        super().__init__(path)
        self.center_map = {1: 'CHUM', 2: 'CHUP', 3: 'CHUS', 6: 'HGJ', 7: 'HMR', 5: 'MDA', 8: 'USZ'}

        self.clinical_key_mapping = {
            "Gender": "sex",
            "Performance Status": "ecog",
            "Tobacco Consumption": "smoking",
            # "Alcohol Consumption": "alcohol",
            "M-stage": "metastasis",
            "HPV Status": "hpv",
            "Treatment": "treatment",
            "Age": "age",
            "dose": "dose",
        }

        self.clinical_encoding = {
            "Gender": {0: 0, 1: 1}, # 0 = female / 1 = male

            "Performance Status": {"ECOG 0": 0, "ECOG 1": 0, "ECOG 2": 1, "ECOG 3": 2, "ECOG 4": 3},

            "Tobacco Consumption": {"Yes": 1, "No": 0},

            # "Alcohol Consumption": {"No": 0, "Yes": 1},

            "M-stage": {"M0": 0, "M1": 1},

            "HPV Status": {"Positive": 1, "Negative": 0},

            "Treatment": {"RT+CH": 1, "RT": 0},
        }

    def get_avg_dose_scipy(self, ct, mask, dose):
        dose_img = sitk.DICOMOrient(sitk.ReadImage(dose), 'RAS')
        mask_img = sitk.DICOMOrient(sitk.ReadImage(mask), 'RAS')
        ct_img = sitk.DICOMOrient(sitk.ReadImage(ct), 'RAS')

        moving_shape = np.array(sitk.GetArrayFromImage(dose_img).shape)
        reference_shape = np.array(sitk.GetArrayFromImage(ct_img).shape)
        zoom = reference_shape / moving_shape
        resample_img = scipy.ndimage.zoom(input=sitk.GetArrayFromImage(dose_img), zoom=zoom, mode="nearest")

        mask_bool = sitk.GetArrayFromImage(mask_img)
        gtv_dose = resample_img[mask_bool == 1].flatten()
        return gtv_dose.mean()

    def build_patients(self, log=None):
        data = {}

        # task 2
        list_of_patients =  glob.glob(os.path.join(self.path, "Task 2", "*"))
        for p in tqdm.tqdm(list_of_patients):
            if not os.path.isdir(p):
                continue

            id = os.path.split(p)[1]

            try:
                ct = dicom_class.CT(glob.glob(os.path.join(p, "*CT*"))[0])
            except IndexError:
                if log: log.warning(f"WARNING: no CT found for patient {id}")
                ct = None

            try:
                dose = dicom_class.CT(glob.glob(os.path.join(p, "*RTDOSE*"))[0])
                if ct: ct.add_rtdose(dose)
            except IndexError:
                if log: log.warning(f"WARNING: no RTDOSE found for patient {id}")
                dose = None

            # calculate GTV dose
            ct = os.path.join(self.path, "Task 1", id, f"{id}__CT.nii.gz")
            mask = os.path.join(self.path, "Task 1", id, f"{id}.nii.gz")
            dose = os.path.join(self.path, "Task 2", id, f"{id}__RTDOSE.nii.gz")
            if os.path.exists(ct) and os.path.exists(mask) and os.path.exists(dose):
                try:
                    dose = int(self.get_avg_dose_scipy(ct, mask, dose))
                    if dose < 30: dose = None
                except IndexError:
                    dose = None
            else:
                dose = None

            df = pandas.read_csv(os.path.join(self.path, "Task 2", "HECKTOR_2025_Training_Task_2.csv"))
            clinical = df[df["PatientID"] == str(id)].to_dict(orient="records")[0]
            clinical["CenterID"] = self.center_map[clinical["CenterID"]]
            clinical.update({"dose": dose})

            p = dicom_class.Patient(patient_id=id, ct=[ct], clinical=clinical)
            data.update({p.id: p})

        # task 3
        list_of_patients =  glob.glob(os.path.join(self.path, "Task 3", "*"))
        for p in tqdm.tqdm(list_of_patients):
            if not os.path.isdir(p):
                continue

            id = os.path.split(p)[1]

            try:
                ct = dicom_class.CT(glob.glob(os.path.join(p, "*CT*"))[0])
                if log: log.warning(f"WARNING: no CT found for patient {id}")
            except IndexError:
                ct = None

            df = pandas.read_csv(os.path.join(self.path, "Task 3", "HECKTOR_2025_Training_Task_3.csv"))
            clinical = df[df["PatientID"] == str(id)].to_dict(orient="records")[0]
            clinical["CenterID"] = self.center_map[clinical["CenterID"]]

            if id in data.keys():
                if not(data[id].ct) and ct:
                    data[id].ct = ct
                
                data[id].clinical.update(clinical)
            else:
                p = dicom_class.Patient(patient_id=id, ct=[ct], clinical=clinical)
                data.update({p.id: p})
        
        return list(data.values())
    
    def generate_clinical_df(self):
        col = {
            "PatientID": "PatientID",
            "Age": "Age",
            "Gender": "Gender",
            "Tobacco Consumption": "Tobacco Consumption",
            "Alcohol Consumption": "Alcohol Consumption",
            "Performance Status": "Performance Status",
            "M-stage": "M-stage",
            "HPV Status": "HPV Status",
        }

        df1 = pandas.read_csv(os.path.join(self.path, "Task 1", "HECKTOR_2025_Training_Task_1.csv"))
        df2 = pandas.read_csv(os.path.join(self.path, "Task 2", "HECKTOR_2025_Training_Task_2.csv"))
        df3 = pandas.read_csv(os.path.join(self.path, "Task 3", "HECKTOR_2025_Training_Task_3.csv"))
        df = df1.merge(df2, how='outer').merge(df3, how='outer')

        df = df[[c for c in df.columns if c in col.keys()]]
        df = df.rename(columns=col, errors="raise")
        return df
    
    def build_dataset(self, path_=None):
        """
        Return features and labels as pandas.Dataframe
        """
        base_path = pathlib.Path("..") if path_ is None else pathlib.Path(path_)

        with open(base_path.joinpath("pickle datasets", "hecktor.pickle"), "rb") as f:
            patients = pickle.load(f)

        # concatenate features and modalities
        features = []
        for file_ in base_path.joinpath("features", "hecktor").iterdir():
            fts = pandas.read_csv(file_)
            fts["modality"] = "clinical" if file_.name.startswith("llm") else "image"
            fts["features"] = file_.stem
            features.append(fts)        
        features = pandas.concat(features)

        # build labels
        label = []
        clinical = []
        for id_, p in patients.items():
            try:
                rfs = p.clinical["RFS"]
                r2y = int(rfs > 2*365)
                r5y = int(rfs > 5*365)
            except KeyError:
                r2y, r5y = None, None

            center = p.clinical["CenterID"]
            center = self.center_map[center] if center in self.center_map.keys() else center
            label.append({"patient": id_, "center": center, recurrence_2y_name: r2y, recurrence_5y_name: r5y})

            # build clinical features
            for k, v in p.clinical.items():
                if k in self.clinical_key_mapping.keys():
                    if k in self.clinical_encoding.keys(): # categorical feature
                        try:
                            v = self.clinical_encoding[k][v]
                        except KeyError:
                            v = None
                    else: # numerical feature
                        try:
                            v = float(v)
                        except ValueError:
                            v = None
                    clinical.append({"patient": id_, "modality": "clinical", "features": self.clinical_key_mapping[k], "name": 0, "value": v})

        label = pandas.DataFrame(label)
        features = pandas.concat([features, pandas.DataFrame(clinical)])
        return features, label

    def get_spatial_transforms():
        tr = [Flip(spatial_axis=-1), SpatialCrop(roi_start=(0,100,0), roi_end=(500,500,300)),]
        return tr


class TCIA(BaseLoader):
    def __init__(self, path):
        super().__init__(path)

    def build_patients(self, log=None):
        data = []
        list_of_patients =  glob.glob(os.path.join(self.path, "manifest*", "*", "*"))
        for p in tqdm.tqdm(list_of_patients):
            if not os.path.isdir(p):
                continue

            id = pathlib.Path(p).name

            # load every DICOM data of patient folder
            patient_data = self.parse_dicom_data(p)

            # group CT with dose
            for rtdose in filter(lambda i: isinstance(i, dicom_class.RTDOSE), patient_data):
                done = False
                for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
                    if rtdose.get_FrameOfReferenceUID() == ct.get_FrameOfReferenceUID():
                        ct.add_rtdose(rtdose, log)
                        done = True
                        break

                if not(done) and log: log.warning(f"WARNING: RTDOSE at {rtdose.path} found no matching CT")

            # group CT with struct
            for rtstruct in filter(lambda i: isinstance(i, dicom_class.RTSTRUCT), patient_data):
                done = False
                for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
                    if rtstruct.get_StudyInstanceUID() == ct.get_StudyInstanceUID():

                        # check for HNSCC-3DCT-RT
                        if ct.rtstruct:
                            dcm = pydicom.dcmread(ct.rtstruct.get_dcm_path())
                            if dcm.get((0x0008, 0x0070)).value == "MIM Software Inc." or dcm.get((0x0008, 0x1090)).value == "MIM":
                                # skip because current RTSTRUCT is preferable
                                continue

                        ct.add_rtstruct(rtstruct, log)
                        done = True
                        break

                if not(done) and log: log.warning(f"WARNING: RTSTRUCT at {rtstruct.path} found no matching CT")

            clinical = self.get_patient_clinical_data(id)
            
            p =  dicom_class.Patient(
                patient_id=id,
                ct=list(filter(lambda i: isinstance(i, dicom_class.CT), patient_data)),
                cbct=list(filter(lambda i: isinstance(i, dicom_class.CBCT), patient_data)),
                clinical=clinical)
        
            data.append(p)
    
        return data
    
    @abstractmethod
    def get_patient_clinical_data(self, patient_id):
        pass


class HeadNeckCTAtlas(TCIA):
    def __init__(self, path):
        super().__init__(path)

    def get_patient_clinical_data(self, patient_id):
        clinical = pandas.read_excel(os.path.join(self.path, "HNSCC-MDA-Data_update_20240514.xlsx"), sheet_name="HNSCC-MDA-Data_update")
        clinical = clinical[clinical["TCIA PatientID"] == patient_id].to_dict('records')
        if clinical: return clinical[0]
        else: return {}

    def generate_clinical_df(self):
        col = {
            "TCIA PatientID": "Patient ID number randomly assigned to each patient prior to anonymizing the DICOM PHI tag (0010,0020)",
            "Sex": "Patient sex, male or female",
            "Age": "Patient age, years",
            "Site": "Primary cancer subsite. CUP = cancer of unknown primary",
            "Histology": "Cancer histopathology. SCC=squamous cell carcinoma",
            "N": "AJCC 7th edition N stage",
            "Stage": "AJCC 7th edition summary stage",
            "RT Total Dose (Gy)": "Total RT dose delivered during radiotherapy.",
            "Dose/Fraction (Gy/fx)": "Dose delivered to prescription target volume (gross disease or post-operative tumor bed)",
            "Number of Fractions": "Number of RT fractions delivered.",
            "Smoking History": "Smoking history coded: 0=never smoker, 1= fewer than 10 pack-years, 2= 10 or more pack-years",
            "Current Smoker": "Current smoker: 0=no, 1=yes",
            "BMI start treat (kg/m2)": "Patient BMI at the start of radiotherapy.",
        }

        df = pandas.read_excel(os.path.join(self.path, "HNSCC-MDA-Data_update_20240514.xlsx"), sheet_name="HNSCC-MDA-Data_update")
        df = df[[c for c in df.columns if c in col.keys()]]
        df = df.rename(columns=col, errors="raise")
        return df
    
    def build_dataset(self, path_=None):
        """
        Return features and labels as pandas.Dataframe
        """
        base_path = pathlib.Path("..") if path_ is None else pathlib.Path(path_)

        with open(base_path.joinpath("pickle datasets", "headneckctatlas.pickle"), "rb") as f:
            patients = pickle.load(f)

        # concatenate features and modalities
        features = []
        for file_ in base_path.joinpath("features", "headneckctatlas").iterdir():
            fts = pandas.read_csv(file_)
            fts["modality"] = "clinical" if file_.name.startswith("llm") else "image"
            fts["features"] = file_.stem
            features.append(fts)        
        features = pandas.concat(features)

        # build labels
        label = []
        for id_, p in patients.items():
            try:
                rfs = p.clinical["Disease-free interval (months)"]
                r2y = int(rfs > 2*12)
                r5y = int(rfs > 5*12)
            except KeyError:
                r2y, r5y = None, None

            label.append({"patient": id_, "center": "MDA", recurrence_2y_name: r2y, recurrence_5y_name: r5y})
        label = pandas.DataFrame(label)
        return features, label

    def get_spatial_transforms():
        tr = [Flip(spatial_axis=-1), SpatialCrop(roi_start=(0,50,0), roi_end=(512,512,350)),]
        return tr


class HeadNeckPETCT(TCIA):
    def __init__(self, path):
        super().__init__(path)

    def get_patient_clinical_data(self, patient_id):
        clinical = []
        for i in ("HGJ", "CHUS", "HMR", "CHUM"):
            df = pandas.read_excel(os.path.join(self.path, "INFOclinical_HN_Version2_30may2018.xlsx"), sheet_name=i)
            df["center"] = i
            clinical.append(df)
        clinical = pandas.concat(clinical)
        clinical = clinical[clinical["Patient #"] == patient_id].to_dict('records')
        if clinical: return clinical[0]
        else: return {}

    def generate_clinical_df(self):
        col = {
            "Patient #": "Patient #",
            "Sex": "Sex",
            "Age": "Age",
            "Primary Site": "Primary Site",
            "N-stage": "N-stage",
            "TNM group stage": "TNM group stage",
            "HPV status": "HPV status",
        }

        df = []
        for i in ("HGJ", "CHUS", "HMR", "CHUM"):
            df__ = pandas.read_excel(os.path.join(self.path, "INFOclinical_HN_Version2_30may2018.xlsx"), sheet_name=i)
            df.append(df__)
        df = pandas.concat(df)
        df = df[[c for c in df.columns if c in col.keys()]]
        df = df.rename(columns=col, errors="raise")
        return df
    
    def build_dataset(self, path_=None):
        """
        Return features and labels as pandas.Dataframe
        """
        base_path = pathlib.Path("..") if path_ is None else pathlib.Path(path_)

        with open(base_path.joinpath("pickle datasets", "headneckpetct.pickle"), "rb") as f:
            patients = pickle.load(f)

        # concatenate features and modalities
        features = []
        for file_ in base_path.joinpath("features", "headneckpetct").iterdir():
            fts = pandas.read_csv(file_)
            fts["modality"] = "clinical" if file_.name.startswith("llm") else "image"
            fts["features"] = file_.stem
            features.append(fts)        
        features = pandas.concat(features)

        # build labels
        label = []
        for id_, p in patients.items():
            diag_endrt_dt = p.clinical["Time – diagnosis to end treatment (days)"]
            diag_lr_dt = p.clinical["Time – diagnosis to LR (days)"]
            diag_dm_dt = p.clinical["Time – diagnosis to DM (days)"]
            diag_death = p.clinical["Time – diagnosis to Death (days)"]
            diag_lastfollow_dt = p.clinical["Time – diagnosis to last follow-up (days)"]

            if pandas.isna(diag_endrt_dt):
                r2y = None
                r5y = None
            elif pandas.notna(diag_lr_dt) or pandas.notna(diag_dm_dt):
                valid_dt = tuple(filter(pandas.notna, (diag_lr_dt, diag_dm_dt)))
                endrt_event_dt = min(map(lambda i: i - diag_endrt_dt, valid_dt))
                r2y = int(endrt_event_dt > 2*365)
                r5y = int(endrt_event_dt > 5*365)
            elif pandas.notna(diag_death) or pandas.notna(diag_lastfollow_dt):
                valid_dt = tuple(filter(pandas.notna, (diag_death, diag_lastfollow_dt)))
                endrt_event_dt = min(map(lambda i: i - diag_endrt_dt, valid_dt))
                r2y = 1 if endrt_event_dt > 2*365 else None
                r5y = 1 if endrt_event_dt > 5*365 else None
            else:
                r2y = None
                r5y = None

            label.append({"patient": id_, "center": p.clinical['center'], recurrence_2y_name: r2y, recurrence_5y_name: r5y})
        label = pandas.DataFrame(label)
        return features, label

    def get_spatial_transforms():
        tr = [Flip(spatial_axis=-1), SpatialCrop(roi_start=(0,100,0), roi_end=(512,512,350)),]
        return tr


class HNSCC3DCTRT(TCIA):
    def __init__(self, path):
        super().__init__(path)
        self.CLINICAL_MAPPING = {
            # SEX
            "M": 1,
            "F": 2,

            # CANCER STAGING
            'I': 1,
            'II': 2,
            'IIA': 2,
            'IIB': 2,
            'III': 3,
            'IVA': 4,

            # ECOG
            'ECOG 0': 0,
            'ECOG 0-1': 0,
            'ECOG 0-2': 0,
            'ECOG 0-3': 0,
            'ECOG 1': 1,
            'ECOG 1-2': 1,
        }

    def get_patient_clinical_data(self, patient_id):
        clinical = pandas.read_excel(os.path.join(self.path, "TCIA 3-6M CTCAE grade.xlsx"))
        clinical = clinical[clinical["HN_P"].astype(int) == int(re.findall("\d+", patient_id)[0])].to_dict('records')
        if clinical: return clinical[0]
        else: return {}


class OropharyngealRadiomicsOutcomes(TCIA):
    def __init__(self, path):
        super().__init__(path)

    def get_patient_clinical_data(self, patient_id):
        clinical = pandas.read_csv(os.path.join(self.path, "Radiomics_Outcome_Prediction_in_OPC_ASRM_corrected.csv"))
        clinical = clinical[clinical["TCIA Radiomics dummy ID of To_Submit_Final"] == patient_id].to_dict('records')
        if clinical: return clinical[0]
        else: return {}

    def generate_clinical_df(self):
        col = {
            "TCIA Radiomics dummy ID of To_Submit_Final": "TCIA Radiomics dummy ID of To_Submit_Final",
            "Gender": "Gender",
            "Age at Diag": "Age at Diag",
            "Smoking status": "Smoking status",
            "Smoking status (Packs-Years)": "Smoking status (Packs-Years)",
            "Cancer subsite of origin": "Cancer subsite of origin",
            "HPV Status": "HPV Status",
            "N-category": "N-category",
            "AJCC Stage (7th edition)": "AJCC Stage (7th edition)",
            "Total prescribed Radiation treatment dose": "Total prescribed Radiation treatment dose",    
        }

        df = pandas.read_csv(os.path.join(self.path, "Radiomics_Outcome_Prediction_in_OPC_ASRM_corrected.csv"))

        df = df[[c for c in df.columns if c in col.keys()]]
        df = df.rename(columns=col, errors="raise")
        return df
    
    def build_dataset(self, path_=None):
        """
        Return features and labels as pandas.Dataframe
        """
        base_path = pathlib.Path("..") if path_ is None else pathlib.Path(path_)

        with open(base_path.joinpath("pickle datasets", "oropharyngealradiomicsoutcomes.pickle"), "rb") as f:
            patients = pickle.load(f)

        # concatenate features and modalities
        features = []
        for file_ in base_path.joinpath("features", "oropharyngealradiomicsoutcomes").iterdir():
            fts = pandas.read_csv(file_)
            fts["modality"] = "clinical" if file_.name.startswith("llm") else "image"
            fts["features"] = file_.stem
            features.append(fts)        
        features = pandas.concat(features)

        # build labels
        label = []
        for id_, p in patients.items():
            try:
                if any([p.clinical[k] == "No" for k in ["Local control", "Regional control", "Freedom from distant metastasis"]]):
                    control_duration_pairs = [
                        (p.clinical["Local control"], p.clinical["Local control_duration of Merged updated ASRM V2"]),
                        (p.clinical["Regional control"], p.clinical["Regional control_duration of Merged updated ASRM V2"]),
                        (p.clinical["Freedom from distant metastasis"], p.clinical["Freedom from distant metastasis_duration of Merged updated ASRM V2"]),
                        ]
                    control_duration_pairs = tuple(filter(lambda i: i[0] == "No" , control_duration_pairs))
                    endrt_event_dt = min(map(lambda i: i[1] - p.clinical["Radiation Treatment_duration"], control_duration_pairs))
                    r2y = int(endrt_event_dt > 2*365)
                    r5y = int(endrt_event_dt > 5*365)
                elif p.clinical["Relapse-free survival"] == "Yes":
                    endrt_event_dt = p.clinical["Days to last FU"] - p.clinical["Radiation Treatment_duration"]
                    r2y = 1 if endrt_event_dt > 2*365 else None
                    r5y = 1 if endrt_event_dt > 5*365 else None
                else:
                    r2y = None
                    r5y = None
            except KeyError:
                r2y = None
                r5y = None

            label.append({"patient": id_, "center": "MDA", recurrence_2y_name: r2y, recurrence_5y_name: r5y})
        label = pandas.DataFrame(label)
        return features, label

    def get_spatial_transforms():
        tr = [Flip(spatial_axis=-1), SpatialCrop(roi_start=(0,0,0), roi_end=(512,512,300)),]
        return tr


class QINHEADNECK(TCIA):
    def __init__(self, path):
        super().__init__(path)

    def get_patient_clinical_data(self, patient_id):
        df1 = pandas.read_excel(os.path.join(self.path, "Batch_01-and-Batch_02-Clinical-Data_aug242020.xlsx"), header=0, sheet_name="uiowa_clinical_data_batch1_aug2")
        df2 = pandas.read_excel(os.path.join(self.path, "Batch_01-and-Batch_02-Clinical-Data_aug242020.xlsx"), header=0, sheet_name="uiowa_clinical_data_batch2_aug2")
        clinical = pandas.concat((df1, df2)).drop(index=0)
        clinical = clinical[clinical["PatientID"] == patient_id].to_dict('records')
        if clinical: return clinical[0]
        else: return {}

    def generate_clinical_df(self):
        col = {
            "PatientID": "PatientID",
            "Patient's Birth Date": "Patient's age",
            "Patient's Sex": "Patient's Sex",
            "Patient's Weight": "Patient's Weight",
            "Patient's Height": "Patient's Height",
            "History of Diabetes Mellitus": "History of Diabetes Mellitus",
            "Alcohol consumption": "Alcohol consumption",
            "Tobacco Smoking Behavior": "Tobacco Smoking Behavior",
            "N Stage": "N Stage",
            "Tumor staging": "Tumor staging",
            "Primary tumor site": "Primary tumor site",
            "Radiotherapy Procedure, Total radiation dose delivered": "Radiotherapy Procedure, Total radiation dose delivered",
            "Radiotherapy Procedure, Radiation dose per fraction": "Radiotherapy Procedure, Radiation dose per fraction",
        }

        df1 = pandas.read_excel(os.path.join(self.path, "Batch_01-and-Batch_02-Clinical-Data_aug242020.xlsx"), header=0, sheet_name="uiowa_clinical_data_batch1_aug2")
        df2 = pandas.read_excel(os.path.join(self.path, "Batch_01-and-Batch_02-Clinical-Data_aug242020.xlsx"), header=0, sheet_name="uiowa_clinical_data_batch2_aug2")
        df = pandas.concat((df1, df2)).drop(index=0)

        # convert birth date to age based on diagnostic date
        df["Patient's Birth Date"] = pandas.to_datetime(df["Diagnositic Procedure, Biopsy, Date of procedure"]).dt.year - df["Patient's Birth Date"]

        df = df[[c for c in df.columns if c in col.keys()]]
        df = df.rename(columns=col, errors="raise")
        return df
    
    def build_dataset(self, path_=None):
        """
        Return features and labels as pandas.Dataframe
        """
        base_path = pathlib.Path("..") if path_ is None else pathlib.Path(path_)

        with open(base_path.joinpath("pickle datasets", "qinheadneck.pickle"), "rb") as f:
            patients = pickle.load(f)

        # concatenate features and modalities
        features = []
        for file_ in base_path.joinpath("features", "qinheadneck").iterdir():
            fts = pandas.read_csv(file_)
            fts["modality"] = "clinical" if file_.name.startswith("llm") else "image"
            fts["features"] = file_.stem
            features.append(fts)        
        features = pandas.concat(features)

        # build labels
        label = []
        for id_, p in patients.items():
            try:
                rtend = p.clinical["Radiotherapy Procedure, Date treatment stopped"]
                if pandas.isna(rtend):
                    r2y = None
                    r5y = None
                else:
                    try:
                        if pandas.notna(p.clinical["Date of cancer recurrence"]):
                            endrt_event_dt = (p.clinical["Date of cancer recurrence"] - rtend).days
                            r2y = int(endrt_event_dt > 2*365)
                            r5y = int(endrt_event_dt > 5*365)
                        else:
                            endrt_event_dt = (p.clinical["Follow-up visit date"] - rtend).days
                            r2y = 1 if endrt_event_dt > 2*365 else None
                            r5y = 1 if endrt_event_dt > 5*365 else None
                    except (KeyError, TypeError):
                        r2y = None
                        r5y = None
            except KeyError:
                r2y = None
                r5y = None

            label.append({"patient": id_, 
                          "center": "UIHC", # University of Iowa Hospital and Clinics
                          recurrence_2y_name: r2y, 
                          recurrence_5y_name: r5y})
        label = pandas.DataFrame(label)
        return features, label

    def get_spatial_transforms():
        tr = [Flip(spatial_axis=-1), SpatialCrop(roi_start=(0,0,0), roi_end=(512,512,300)),]
        return tr


class RADCURE(TCIA):
    def __init__(self, path):
        super().__init__(path)

        self.clinical_key_mapping = {
            "Sex": "sex",
            "ECOG PS": "ecog",
            "Smoking Status": "smoking",
            "Stage": "stage",
            "HPV": "hpv",
            "Tx Modality": "treatment",
            "Age": "age",
            "Dose": "dose",
        }

        self.clinical_encoding = {
            "Sex": {"Female": 0, "Male": 1},
            
            "ECOG PS": {"ECOG 0": 0, "ECOG 2": 1, "ECOG 1": 0, "ECOG 4": 3, "ECOG 3": 2, "ECOG-Pt 2": 1, 
                        "ECOG 0-1": 0, "ECOG-Pt 0": 0, "ECOG-Pt 1": 0},
            
            "Smoking Status": {"Ex-smoker": 1, "Non-smoker": 0, "Current": 2},
            
            "Stage": {"IVB": 4, "I": 1, "IVA": 4, "III": 3, "II": 2, "IV": 4, "IIIC": 3, "IB": 1, 
                      "IIA": 2, "IIIA": 3, "IVC": 4, "IIB": 2, "0": 0},
            
            "HPV": {"Yes, Negative": 0, "Yes, positive": 1},
            
            "Tx Modality": {"RT alone": 0, "ChemoRT": 1, "ChemoRT ": 1, "Postop RT alone": 0},
        }

    def get_patient_clinical_data(self, patient_id):
        clinical = pandas.read_excel(os.path.join(self.path, "RADCURE_Clinical_v04_20241219 (1).xlsx"))
        clinical = clinical[clinical["patient_id"] == patient_id].to_dict('records')
        if clinical: return clinical[0]
        else: return {}

    def generate_clinical_df(self):
        col = {
            "patient_id": "Patient ID number randomly assigned to each patient prior to anonymizing the DICOM PHI tag (0010,0020)",
            "Age": "Patient age, years",
            "Sex": "Patient sex, male or female",
            "ECOG PS": "ECOG Performance status scale - assessment of patient's functional status;GRADE	ECOG PERFORMANCE STATUS;0= Fully active, able to carry on all pre-disease performance without restriction;1=Restricted in physically strenuous activity but ambulatory and able to carry out work of a light or sedentary nature, e.g., light house work, office work;2=Ambulatory and capable of all selfcare but unable to carry out any work activities; up and about more than 50% of waking hours;3=Capable of only limited selfcare; confined to bed or chair more than 50% of waking hours;4=Completely disabled; cannot carry on any selfcare; totally confined to bed or chair;5=Dead",
            "Smoking PY": "Number of packs smoked in a year",
            "Smoking Status": "Smoking status at first consultation:Current smoker, ex-smoker, non-smoker",
            "Ds Site": "Primary cancer site",
            "Subsite": "Primary cancer subsite",
            "N": "AJCC 7th edition N category ",
            "Stage": "AJCC 7th edition stage groups",
            "Path": "Pathologic diagnosis/histology type",
            "HPV": "Tumor HPV status determined by p16 IHC +/- HPV DNA by PCR. Blank cell indicates no data available.",
            "Tx Modality": "How the surgery, radiation, and chemotherapy are combined - see mapping terminology",
            "Chemo": "Yes=received concurrent chemoradiotherapy, none=did not receive concurrent chemoradiotherapy",
            "Dose": "Total RT dose delivered during radiotherapy.",
            "Fx": "Number of RT fractions delivered."}

        df = pandas.read_excel(os.path.join(self.path, "RADCURE_Clinical_v04_20241219 (1).xlsx"))
        df = df[[c for c in df.columns if c in col.keys()]]
        df = df.rename(columns=col, errors="raise")
        return df
    
    def build_dataset(self, path_=None):
        """
        Return features and labels as pandas.Dataframe
        """
        base_path = pathlib.Path("..") if path_ is None else pathlib.Path(path_)

        with open(base_path.joinpath("pickle datasets", "radcure.pickle"), "rb") as f:
            patients = pickle.load(f)

        # concatenate features and modalities
        features = []
        for file_ in base_path.joinpath("features", "radcure").iterdir():
            if file_.name.startswith("llm"):
                continue
            fts = pandas.read_csv(file_)
            fts["modality"] = "image"
            fts["features"] = file_.stem
            features.append(fts)
        features = pandas.concat(features)

        # build labels
        label = []
        clinical = []
        for id_, p in patients.items():
            try:
                # add fractions to rt start date to obtain RT end date
                rtend = p.clinical["RT Start"].date() + datetime.timedelta(days=int(p.clinical["Fx"] / 5)*7)

                event_dt = tuple(filter(pandas.notna, [p.clinical[k] for k in ["Date Local", "Date Regional", "Date Distant"]]))
                if event_dt:
                    event_dt = event_dt[0] if len(event_dt) == 1 else min(*event_dt)
                    endrt_event_dt = (event_dt.date() - rtend).days
                    r2y = int(endrt_event_dt > 2*365)
                    r5y = int(endrt_event_dt > 5*365)
                else:
                    if pandas.notna(p.clinical["Last FU"]):
                        endrt_event_dt = (p.clinical["Last FU"].date() - rtend).days
                        r2y = 1 if endrt_event_dt > 2*365 else None
                        r5y = 1 if endrt_event_dt > 5*365 else None
                    else:
                        r2y = None
                        r5y = None
            except ValueError:
                r2y = None
                r5y = None

            label.append({"patient": id_, 
                          "center": "UHN", # University Health Network
                          recurrence_2y_name: r2y, 
                          recurrence_5y_name: r5y})
            
            # build clinical features
            for k, v in p.clinical.items():
                if k in self.clinical_key_mapping.keys():
                    if k in self.clinical_encoding.keys(): # categorical feature
                        try:
                            v = self.clinical_encoding[k][v]
                        except KeyError:
                            v = None
                    else: # numerical feature
                        try:
                            v = float(v)
                        except ValueError:
                            v = None
                    clinical.append({"patient": id_, "modality": "clinical", "features": self.clinical_key_mapping[k], "name": 0, "value": v})
        
        label = pandas.DataFrame(label)
        features = pandas.concat([features, pandas.DataFrame(clinical)])
        return features, label

    def get_spatial_transforms():
        tr = [Flip(spatial_axis=-1), SpatialCrop(roi_start=(0,0,0), roi_end=(512,512,300)),]
        return tr

cohorts_map = {
    "artix": ARTIX,
    "hecktor": HECKTOR,
    "headneckctatlas": HeadNeckCTAtlas,
    "headneckpetct": HeadNeckPETCT,
    "hnscc3dctrt": HNSCC3DCTRT,
    "oropharyngealradiomicsoutcomes": OropharyngealRadiomicsOutcomes,
    "qinheadneck": QINHEADNECK,
    "radcure": RADCURE,
}
