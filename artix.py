import os, glob, pathlib
import pydicom
import pandas

import dicom_class
import dicom_utils

# (0008, 0060) Modality
# 'CT', 'RTSTRUCT', 'RTDOSE'
# CT vs CBCT: (0008, 0070) Manufacturer of CBCT is 'ELEKTA'
# threshold for xerostomia is <500 mg/min

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

    # print(f"{len(data)} DICOM loaded from {path}")
    return data

def load_patient(path, id_map, patient_description_csv=None, efficacy_csv=None, salivary_csv=None):
    """
    Load a patient folder and return images with dose and rtplan

    Args:
        path (str) path to patient folder
        id_map (str) path to CSV mapping folder IDs to clinical data IDs
        patient_description_csv (str) path to patient description CSV
        efficacy_csv (str) path to efficacy data CSV
        salivary_csv (str) path to salivary data CSV

    return Patient object
    """

    id_map = pandas.read_excel(id_map)
    id = pathlib.Path(path).name

    # convert patient ID from folder to clinical data
    id = int(id_map[id_map["My Identifier ID"] == float(id)]["USUBJID"].item())
    
    # load every DICOM data of patient folder
    patient_data = load_folder(path)

    # group CT with dose
    for rtdose in filter(lambda i: isinstance(i, dicom_class.RTDOSE), patient_data):
        done = False
        for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
            if rtdose.get_FrameOfReferenceUID() == ct.get_FrameOfReferenceUID() or \
                rtdose.get_StudyInstanceUID() == ct.get_StudyInstanceUID() or \
                    max(dicom_utils.get_directory_level(rtdose.path, ct.path)) < 2 and ct.rtdose is None:
                ct.add_rtdose(rtdose)
                done = True
                break

        if not done:
            print(f"WARNING: RTDOSE at {rtdose.path} found no matching CT")

    # group CT with struct
    for rtstruct in filter(lambda i: isinstance(i, dicom_class.RTSTRUCT), patient_data):
        done = False
        for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
            if rtstruct.get_FrameOfReferenceUID() == ct.get_FrameOfReferenceUID() or \
                rtstruct.get_StudyInstanceUID() == ct.get_StudyInstanceUID() or \
                    max(dicom_utils.get_directory_level(rtstruct.path, ct.path)) < 2 and ct.rtstruct is None:
                ct.add_rtstruct(rtstruct)
                done = True
                break

        if not done:
            print(f"WARNING: RTSTRUCT at {rtstruct.path} found no matching CT")

    # load clinical data
    patient_clinical = dict()

    if not patient_description_csv is None:
        df = pandas.read_csv(patient_description_csv, sep=";")
        if not id in df["USUBJID"].unique():
            print(f"WARNING: patient folder id {id} not found in patient description CSV at {patient_description_csv}")
        else:
            df = df[df["USUBJID"] == id]
            patient_clinical.update({
                "age": df["AGE"].item(), 
                "sexe": df["SEX"].item(), 
                "hpv": df["ST_HPV"].item(),
                "arm": df["ACTARMCD"].item(),
                "incldt": df["INCDT"].item(),
                "randdt": df["RANDT"].item(),
                })

    if not efficacy_csv is None:
        df = pandas.read_csv(efficacy_csv, sep=";")
        if not id in df["USUBJID"].unique():
            print(f"WARNING: patient folder id {id} not found in patient description CSV at {efficacy_csv}")
        else:        
            df = df[df["USUBJID"] == id]
            patient_clinical.update({
                "prog_recc": df['PROG'].item(),
                "progdt": df['PROGDT'].item(),
                })
    
    if not salivary_csv is None:
        df = pandas.read_csv(salivary_csv, sep=";")
        if not id in df["USUBJID"].unique():
            print(f"WARNING: patient folder id {id} not found in patient description CSV at {salivary_csv}")
        else:
            df = df[(df["USUBJID"] == id) & (df["MEASTYP"] == "Stimulated salivation flow")]
            for _, row in df.iterrows():
                if isinstance(row["MEAS_VAL"], (int, float)) or row["MEAS_VAL"].isdigit():
                    patient_clinical.update({f"{row['MEASTYP']} ({row['VISITID']})": float(row["MEAS_VAL"])})
                else:
                    patient_clinical.update({f"{row['MEASTYP']} ({row['VISITID']})": None})
    
    return dicom_class.Patient(
        id_=id,
        ct=list(filter(lambda i: isinstance(i, dicom_class.CT), patient_data)),
        cbct=list(filter(lambda i: isinstance(i, dicom_class.CBCT), patient_data)),
        clinical=patient_clinical,
    )
