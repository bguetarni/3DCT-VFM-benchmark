import logging
import os, glob, pathlib
import pydicom
import pandas

import dicom_class
import dicom_utils

# (0008, 0060) Modality
# 'CT', 'RTSTRUCT', 'RTDOSE'
# CT vs CBCT: (0008, 0070) Manufacturer of CBCT is 'ELEKTA'
# threshold for xerostomia is <500 mg/min

def salivation_flow_parsing(df):
    df = df[df["MEASTYP"] == "Stimulated salivation flow"]
    result = {f"{row['MEASTYP']} ({row['VISITID']})": row["MEAS_VAL"] for _, row in df.iterrows()}
    return result

def mean_dose_parsing(df):
    df = df[df["DOSISEQ"] == "Initial"]
    keys = ["PAROH_DOSE", "PAROC_DOSE", "SMAXH_DOSE", "SMAXC_DOSE", "MOUTH_DOSE"]
    result = {k: df[k].unique()[0] for k in keys}
    return result

CLINICAL_KEYS = {
    "age": "AGE",
    "sexe": "SEX",
    "hpv": "ST_HPV",
    "arm": "ACTARMCD",
    "inclusion_dt": "INCDT",
    "randomization_dt": "RANDT",
    "is_prog_recc": 'PROG',
    "progression_dt": 'PROGDT',
    "salivation_flow_parsing": salivation_flow_parsing,
    "start_treatment_dt": "RTSTDT",
    "mean_dose": mean_dose_parsing,
}

def load_folder(path):
    """
    Load a folder which could be anything (collection of imaging, CT, RTDOSE, ...)

    return list of objects of type (CT, CBCT, RTDOSE, RTSTRUCT)
    """

    data = []
    try:
        if all([pydicom.misc.is_dicom(j) for j in glob.glob(os.path.join(path, "*"))]):
            # it is DICOM folder
            
            if len(os.listdir(path)) == 0:
                return []
            
            dcm = pydicom.dcmread(os.path.join(path, os.listdir(path)[0]))
            type = dcm.get((0x0008, 0x0060)).value
            if type == "CT":
                if dcm.get((0x0008, 0x0070)).value == "ELEKTA":
                    return [dicom_class.CBCT(path)]
                else:
                    return [dicom_class.CT(path)]
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
            data.extend(load_folder(folder))

    return data

def load_patient(path, id_map, clinical_csv=None, log=None):
    """
    Load a patient folder and return images with dose and rtplan

    Args:
        path (str) path to patient folder
        id_map (str) path to CSV mapping folder IDs to clinical data IDs
        clinical_csv (List[str]) list of path to clinical data CSV file
        log (str) path to file to log warnings and errors

    return Patient object
    """

    if not log is None:
        logging.basicConfig(
            filename=log,
            format='%(levelname)s: %(message)s',
            filemode='w',
            level=logging.DEBUG)
        log = logging.getLogger()

    id = pathlib.Path(path).name

    # convert patient ID from folder to clinical data
    id_map = pandas.read_excel(id_map)
    id = int(id_map[id_map["My Identifier ID"] == float(id)]["USUBJID"].item())
    
    # load every DICOM data of patient folder
    patient_data = load_folder(path)

    # print number of DICOM data loaded
    data_type = set(map(type, patient_data))
    data_type = {k: len(list(filter(lambda i: isinstance(i, k), patient_data))) for k in data_type}
    print(f"total data loaded from {path}")
    for k, v in data_type.items():
        print(f"{k}:\t {v}")

    # group CT with dose
    for rtdose in filter(lambda i: isinstance(i, dicom_class.RTDOSE), patient_data):
        done = False
        for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
            if rtdose.get_FrameOfReferenceUID() == ct.get_FrameOfReferenceUID() or \
                rtdose.get_StudyInstanceUID() == ct.get_StudyInstanceUID() or \
                    max(dicom_utils.get_directory_level(rtdose.path, ct.path)) < 2 and ct.rtdose is None:
                ct.add_rtdose(rtdose, log)
                done = True
                break

        if not done and not log is None:
            log.warning(f"WARNING: RTDOSE at {rtdose.path} found no matching CT")

    # group CT with struct
    for rtstruct in filter(lambda i: isinstance(i, dicom_class.RTSTRUCT), patient_data):
        done = False
        for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
            if rtstruct.get_FrameOfReferenceUID() == ct.get_FrameOfReferenceUID() or \
                rtstruct.get_StudyInstanceUID() == ct.get_StudyInstanceUID() or \
                    max(dicom_utils.get_directory_level(rtstruct.path, ct.path)) < 2 and ct.rtstruct is None:
                ct.add_rtstruct(rtstruct, log)
                done = True
                break

        if not done and log is None:
            log.warning(f"WARNING: RTSTRUCT at {rtstruct.path} found no matching CT")

    # load clinical data
    patient_clinical = dict()
    if not clinical_csv is None:

        # load and merge all clinical CSV files
        df = None
        for p in clinical_csv:
            sub_df = pandas.read_csv(p, sep=";", encoding='ISO-8859-1')
            if df is None:
                df = sub_df
            else:
                # merge DataFrames based on common columns
                common_columns = set(df.columns).intersection(set(sub_df.columns))
                df = pandas.merge(df, sub_df, on=list(common_columns), how='inner')
        
        # convert USUBJID
        df["USUBJID"] = df["USUBJID"].astype(int)

        # if patient ID available gather clinical data
        if not id in df["USUBJID"].unique() and log is None:
            log.warning(f"WARNING: patient folder id {id} not found in clinical data")
        else:
            df = df[df["USUBJID"] == id]

            # populate patient clinical data based on CLINICAL_KEYS
            for k, v in CLINICAL_KEYS.items():
                if callable(v):
                    patient_clinical.update(v(df))
                else:
                    patient_clinical.update({k: df[v].unique()[0]})
    
    return dicom_class.Patient(
        id_=id,
        ct=list(filter(lambda i: isinstance(i, dicom_class.CT), patient_data)),
        cbct=list(filter(lambda i: isinstance(i, dicom_class.CBCT), patient_data)),
        clinical=patient_clinical,
    )


class ARTIX:
    def __init__(self, path):
        self.path = path
        self.patients = []

    def add_patient(self, p):
        self.patients.append(p)

    def get_relative_path(self, subpath):
        """
        Return relative path to this dataset
        """
        return os.path.relpath(self.path, start=subpath)
