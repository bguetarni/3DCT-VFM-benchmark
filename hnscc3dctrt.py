import os, glob, pathlib, logging, re
import pydicom
import pandas

import dicom_class


def load_folder(path):
    """
    Load a folder which could be anything (collection of imaging, CT, RTDOSE, ...)

    return list of objects of type (CT, RTDOSE, RTSTRUCT)
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

def load_patient(path, clinical=None, log=None, **kwargs):
    """
    Load a patient folder and return images with dose and rtplan

    Args:
        path (str) path to patient folder
        id_map (str) path to CSV mapping folder IDs to clinical data IDs
        clinical (str) path to clinical fodler with csv files
        log (str) path to file to log warnings and errors

    return Patient object
    """

    if not(log is None):
        logging.basicConfig(
            filename=log,
            format='%(levelname)s: %(message)s',
            filemode='w',
            level=logging.DEBUG)
        log = logging.getLogger()

    id = pathlib.Path(path).name

    # load every DICOM data of patient folder
    patient_data = load_folder(path)

    # print number of DICOM data loaded
    data_type = {k: len(list(filter(lambda i: isinstance(i, k), patient_data))) for k in set(map(type, patient_data))}
    print(f"total data loaded from {path}")
    for k, v in data_type.items():
        print(f"{k}:\t {v}")

    # group CT with dose
    for rtdose in filter(lambda i: isinstance(i, dicom_class.RTDOSE), patient_data):
        done = False
        for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
            if rtdose.get_FrameOfReferenceUID() == ct.get_FrameOfReferenceUID():
                ct.add_rtdose(rtdose, log)
                done = True
                break

        if not(done) and not(log is None):
            log.warning(f"WARNING: RTDOSE at {rtdose.path} found no matching CT")

    # group CT with struct
    for rtstruct in filter(lambda i: isinstance(i, dicom_class.RTSTRUCT), patient_data):
        done = False
        for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
            if rtstruct.get_StudyInstanceUID() == ct.get_StudyInstanceUID():
                ct.add_rtstruct(rtstruct, log)
                done = True
                break

        if not(done) and not(log is None):
            log.warning(f"WARNING: RTSTRUCT at {rtstruct.path} found no matching CT")

    # load clinical data
    if not(clinical is None):
        clinical_d = pandas.read_excel(clinical)
        clinical_d = clinical_d[clinical_d["HN_P"] == int(re.findall("\d+", id)[0])].to_dict('records')[0]
    
    return dicom_class.Patient(
        id_=id,
        ct=list(filter(lambda i: isinstance(i, dicom_class.CT), patient_data)),
        cbct=list(filter(lambda i: isinstance(i, dicom_class.CBCT), patient_data)),
        clinical=clinical_d,
    )
