from abc import ABC, abstractmethod
import glob, os, tqdm, pathlib, re
import pydicom
import pandas

import dicom_class
import dicom_utils


class BaseLoader(ABC):
    def __init__(self, path):
        self.path = path

    # @abstractmethod
    # def load_clinical(self):
    #     """
    #     age, sex, ecog, cancer stage, xerostomia, hpv, lrr, rfs (days), os (days)
    #     """
    #     pass 

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
            id = str(id_map[id_map["My Identifier ID"].astype(int) == int(id)]["USUBJID"].item()).zfill(3)
            
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


class HECKTOR(BaseLoader):
    def __init__(self, path):
        super().__init__(path)
        self.CLINICAL_MAPPING = {} #TODO

    def build_patients(self, log=None):
        data = {}

        # task 2
        list_of_patients =  glob.glob(os.path.join(self.path, "Task 2", "*"))
        for p in tqdm.tqdm(list_of_patients):
            if not os.path.isdir(p):
                continue

            id = os.path.split(p)[1]

            try:
                ct = dicom_class.CT(glob.glob(os.path.join(p, "*_CT.nii.gz"))[0])
            except IndexError:
                if log: log.warning(f"WARNING: no CT found for patient {id}")
                ct = None

            try:
                dose = dicom_class.CT(glob.glob(os.path.join(p, "*_RTDOSE.nii.gz"))[0])
                if ct:
                    ct.add_rtdose(dose)
            except IndexError:
                if log: log.warning(f"WARNING: no RTDOSE found for patient {id}")
                pass

            df = pandas.read_csv(os.path.join(self.path, "Task 2", "HECKTOR_2025_Training_Task_2.csv"))
            clinical = df[df["PatientID"] == str(id)].to_dict(orient="records")[0]

            p = dicom_class.Patient(patient_id=id, ct=[ct], clinical=clinical)
            data.update({p.id: p})

        # task 3
        list_of_patients =  glob.glob(os.path.join(self.path, "Task 3", "*"))
        for p in tqdm.tqdm(list_of_patients):
            if not os.path.isdir(p):
                continue

            id = os.path.split(p)[1]

            try:
                ct = dicom_class.CT(glob.glob(os.path.join(p, "*_CT.nii.gz"))[0])
                if log: log.warning(f"WARNING: no CT found for patient {id}")
            except IndexError:
                ct = None

            df = pandas.read_csv(os.path.join(self.path, "Task 3", "HECKTOR_2025_Training_Task_3.csv"))
            clinical = df[df["PatientID"] == str(id)].to_dict(orient="records")[0]

            if id in data.keys():
                if not(data[id].ct) and ct:
                    data[id].ct = ct
                
                data[id].clinical.update({clinical})
            else:
                p = dicom_class.Patient(patient_id=id, ct=[ct], clinical=clinical)
                data.update({p.id: p})
        
        return list(data.values())


class TCIA(BaseLoader):
    def __init__(self, path):
        super().__init__(path)

    def build_patients(self, log=None):
        data = []
        list_of_patients =  glob.glob(os.path.join(self.path, "manifest*", "*", "*"))
        for p in tqdm.tqdm(list_of_patients):
            if not os.path.isdir(p):
                continue

            id = pathlib.Path(self.path).name

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

class RADCURE(TCIA):
    def __init__(self, path):
        super().__init__(path)

    def get_patient_clinical_data(self, patient_id):
        clinical = pandas.read_excel(os.path.join(self.path, "RADCURE_Clinical_v04_20241219 (1).xlsx"))
        clinical = clinical[clinical["patient_id"] == patient_id].to_dict('records')
        if clinical: return clinical[0]
        else: return {}
