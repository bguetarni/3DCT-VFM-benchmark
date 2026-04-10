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
from monai.transforms import Flip, SpatialCrop

import dicom_class
import dicom_utils

CATEGORICAL_CLINICAL_VARIABLES = ["sex", "ecog", "smoking", "stage", "hpv", "treatment", 
                                  "surgery", "localisation", "metastasis"]

def convert2date(dt):
    return datetime.datetime.strptime(dt, "%d/%m/%Y").date()


class BaseLoader(ABC):
    def __init__(self, path):
        self.path = path

    def build(self, logger):
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
            files = glob.glob(os.path.join(path, "*"))
            if all(map(pydicom.misc.is_dicom, files)):
                # it is DICOM folder

                if len(files) == 0:
                    return []
                
                dcm = pydicom.dcmread(files[0])
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

    def load_imaging_features(self, dir_path):
        features = {}
        for file_ in dir_path.iterdir():
            with open(file_, "rb") as f:
                fts = pickle.load(f)
            for p in list(fts.keys()):
                fts_p = fts[p]
                p = str(p)   # make sure patient ID is string for consistency
                if not (p in features.keys()):
                    features.update({p: {}})
                features[p].update({file_.stem: fts_p})
        return features


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


class HeadNeckPETCT(TCIA):
    """
    GTV dose according to center
        HGJ - 67.5 Gy
        CHUS - disease specific [50 - 70 Gy]
        HMR - patient specific [66 - 70 Gy]
        CHUM - 70 Gy
    """
    def __init__(self, path):
        super().__init__(path)

        self.clinical_key_mapping = {
            "Age": "age",
            "Sex": "sex",
            "HPV status": "hpv",
            "Primary Site": "localisation",
            "Surgery": "surgery",
            "M-stage": "metastasis",
            "Therapy": "treatment",
            "TNM group stage": "stage",
            "dose": "dose",
        }

        self.clinical_encoding = {
            "sex": {"M": 1, "F": 0, "m": 1},

            "metastasis": {"M0": 0},

            "stage": {"stage III": 3, "stage IIB": 2, "stage IVA": 4, "stage IV": 4, 
                                "stage IVB": 4, "stage II": 2, "stage I": 1, "Stade II": 2, 
                                "Stade III": 3, "Stade IVA": 4, "Stade I": 1, "Stade IVB": 4, 
                                "Stage III": 3, "Stage IVA": 4, "Stage IIB": 2, "Stage II": 2, 
                                "Stage IV": 4, "StageII": 2},
            
            "hpv": {"-": 0, "+": 1},

            "treatment": {"chemo radiation": 1, "radiation": 0},

            "surgery": {"NO": 0, "YES": 1},

            "localisation": {
                # oropharynx
                "Oropharynx": 0,
                # Pharynx & Larynx
                "Larynx": 1, "Nasopharynx": 1, "Hypopharynx": 1,
                # others
                "unknown": 2, "nan": 2}
        }

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
    
    def get_features_labels(self, path_=None):

        base_path = pathlib.Path(".") if path_ is None else pathlib.Path(path_)

        with open(base_path.joinpath("pickle datasets", "headneckpetct.pickle"), "rb") as f:
            patients = pickle.load(f)

        imaging = self.load_imaging_features(base_path.joinpath("features", "headneckpetct"))

        with open(base_path.joinpath("features", "headneckpetct", "radiomics.pkl"), "rb") as f:
            radiomics = pickle.load(f)

        rfs = {}
        clinical = {}
        for id_, p in patients.items():
            diag_endrt_dt = p.clinical["Time – diagnosis to end treatment (days)"]
            diag_lr_dt = p.clinical["Time – diagnosis to LR (days)"]
            diag_dm_dt = p.clinical["Time – diagnosis to DM (days)"]
            diag_death = p.clinical["Time – diagnosis to Death (days)"]
            diag_lastfollow_dt = p.clinical["Time – diagnosis to last follow-up (days)"]

            if pandas.isna(diag_endrt_dt):
                rfs_T = None
                rfs_delta = None
            elif pandas.notna(diag_lr_dt) or pandas.notna(diag_dm_dt):
                valid_dt = tuple(filter(pandas.notna, (diag_lr_dt, diag_dm_dt)))
                endrt_event_dt = min(map(lambda i: i - diag_endrt_dt, valid_dt))
                rfs_T = endrt_event_dt
                rfs_delta = 1
            elif pandas.notna(diag_death) or pandas.notna(diag_lastfollow_dt):
                valid_dt = tuple(filter(pandas.notna, (diag_death, diag_lastfollow_dt)))
                endrt_event_dt = min(map(lambda i: i - diag_endrt_dt, valid_dt))
                rfs_T = endrt_event_dt
                rfs_delta = 0
            else:
                rfs_T = None
                rfs_delta = None

            rfs.update({id_: {"T": rfs_T, "delta": rfs_delta}})

            # build clinical features
            for k, v in p.clinical.items():
                if k in self.clinical_key_mapping.keys():
                    k = self.clinical_key_mapping[k]
                    if k in self.clinical_encoding.keys(): # categorical feature
                        try:
                            v = self.clinical_encoding[k][v]
                        except KeyError:
                            v = None
                    else: # numerical feature
                        try:
                            v = float(v)
                        except (ValueError, TypeError):
                            v = None

                    if not(id_ in clinical.keys()):
                        clinical.update({id_: {}})
                    clinical[id_].update({k: v})

            # add GTV volume (voxels) from radiomics
            if id_ in radiomics.keys():
                clinical[id_]["volume"] = radiomics[id_]["original_shape_VoxelVolume"]
            else:
                clinical[id_]["volume"] = None
        
        return imaging, clinical, rfs

    def build_patients(self, log=None):
        patients = super().build_patients(log)
        print("calculating GTV dose...")
        for p in tqdm.tqdm(patients, ncols=50):
            for ct in p.ct:
                dose = ct.rtdose.get_GTV_dose() if ct.rtdose else None
                p.clinical.update({"dose": dose})
        return patients

    def get_spatial_transforms():
        tr = [Flip(spatial_axis=-1), SpatialCrop(roi_start=(0,100,0), roi_end=(512,512,350)),]
        return tr


class RADCURE(TCIA):
    def __init__(self, path):
        super().__init__(path)

        self.clinical_default_values = {
            "surgery": "NO",   # only 3% of patients had surgery
        }

        self.clinical_key_mapping = {
            "Age": "age",
            "Sex": "sex",
            "Smoking Status": "smoking",
            "HPV": "hpv",
            "ECOG PS": "ecog",
            "Ds Site": "localisation",
            "Tx Modality": "treatment",
            "METASTASIS": "metastasis",
            "Stage": "stage",
            "Dose": "dose",
        }

        self.clinical_encoding = {
            "surgery": {"NO": 0},

            "sex": {"Female": 0, "Male": 1},
            
            "ecog": {"ECOG 0": 0, "ECOG 2": 1, "ECOG 1": 0, "ECOG 4": 3, "ECOG 3": 2, "ECOG-Pt 2": 1, 
                        "ECOG 0-1": 0, "ECOG-Pt 0": 0, "ECOG-Pt 1": 0},
            
            "smoking": {"Ex-smoker": 1, "Non-smoker": 0, "Current": 2},
            
            "stage": {"IVB": 4, "I": 1, "IVA": 4, "III": 3, "II": 2, "IV": 4, "IIIC": 3, "IB": 1, 
                      "IIA": 2, "IIIA": 3, "IVC": 4, "IIB": 2, "0": 0},
            
            "hpv": {"Yes, Negative": 0, "Yes, positive": 1},
            
            "treatment": {"RT alone": 0, "ChemoRT": 1, "ChemoRT ": 1, "Postop RT alone": 0},

            "metastasis": {"M0": 0, "MX": None, "M1": 1},

            "localisation": {
                # oropharynx
                "Oropharynx": 0,
                # Pharynx & Larynx
                "Larynx": 1, "Hypopharynx": 1, "Nasopharynx": 1,
                # others
                "Unknown": 2, "Lip & Oral Cavity": 2, "nasal cavity": 2, "Skin": 2, "Paranasal Sinus": 2, "Sarcoma": 2, 
                "esophagus": 2, "Paraganglioma": 2, "Esophagus": 2, "Nasal Cavity": 2, "benign tumor": 2, "Salivary Glands": 2, 
                "Other": 2, "Orbit": 2, "Lacrimal gland": 2},
        }

        self.localisation_mapping = {0: "Oropharynx", 1: "Pharynx & Larynx", 2: "Others"}

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
    
    def get_features_labels(self, path_=None):

        base_path = pathlib.Path(".") if path_ is None else pathlib.Path(path_)

        with open(base_path.joinpath("pickle datasets", "radcure.pickle"), "rb") as f:
            patients = pickle.load(f)

        imaging = self.load_imaging_features(base_path.joinpath("features", "radcure"))

        with open(base_path.joinpath("features", "radcure", "radiomics.pkl"), "rb") as f:
            radiomics = pickle.load(f)

        rfs = {}
        clinical = {}
        for id_, p in patients.items():
            try:
                # add fractions to rt start date to obtain RT end date
                rtend = p.clinical["RT Start"].date() + datetime.timedelta(days=int(p.clinical["Fx"] / 5)*7)

                event_dt = tuple(filter(pandas.notna, [p.clinical[k] for k in ["Date Local", "Date Regional", "Date Distant"]]))
                if event_dt:
                    event_dt = event_dt[0] if len(event_dt) == 1 else min(*event_dt)
                    endrt_event_dt = (event_dt.date() - rtend).days
                    rfs_T = endrt_event_dt
                    rfs_delta = 1
                else:
                    if pandas.notna(p.clinical["Last FU"]):
                        endrt_event_dt = (p.clinical["Last FU"].date() - rtend).days
                        rfs_T = endrt_event_dt
                        rfs_delta = 0
                    else:
                        rfs_T = None
                        rfs_delta = None
            except ValueError:
                rfs_T = None
                rfs_delta = None

            rfs.update({id_: {"T": rfs_T, "delta": rfs_delta}})
            
            # build clinical features
            clinical.update({id_: {}})
            for k, v in p.clinical.items():
                if k in self.clinical_key_mapping.keys():
                    k = self.clinical_key_mapping[k]
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
                    
                    clinical[id_].update({k: v})
                else:
                    # exception for RADCURE
                    # must include RADCURE challenge split information
                    # do not change or make sure field 'RADCURE-challenge' is included
                    clinical[id_].update({k: v})

            # add default values for clinical features
            for k, v in self.clinical_default_values.items():
                if k in self.clinical_encoding.keys():
                    clinical[id_][k] = self.clinical_encoding[k][v]
                else:
                    clinical[id_][k] = self.clinical_default_values[k]
            
            # add GTV volume (voxels) from radiomics
            if id_ in radiomics.keys():
                clinical[id_]["volume"] = radiomics[id_]["original_shape_VoxelVolume"]
            else:
                clinical[id_]["volume"] = None
        
        return imaging, clinical, rfs

    def get_spatial_transforms():
        tr = [Flip(spatial_axis=-1), SpatialCrop(roi_start=(0,0,0), roi_end=(512,512,300)),]
        return tr

cohorts_map = {
    "headneckpetct": HeadNeckPETCT,
    "radcure": RADCURE,
}
