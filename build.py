import glob
import os
import tqdm
import argparse
import pickle
from datetime import datetime
import pathlib
import logging
import pydicom

import dicom_class

def is_CT_RTSTRUCT_match(ct, rtstruct):
    return (rtstruct.get_StudyID() == ct.get_StudyID() and ct.rtstruct is None) or \
        rtstruct.get_StudyInstanceUID() == ct.get_StudyInstanceUID() or \
            (max(get_directory_level(rtstruct.path, ct.path)) < 2 and ct.rtstruct is None)


def is_CT_RTDOSE_match(ct, rtdose):
    return rtdose.get_FrameOfReferenceUID() == ct.get_FrameOfReferenceUID() or \
                rtdose.get_StudyInstanceUID() == ct.get_StudyInstanceUID() or \
                    ct.rtdose is None


def is_CT(dcm, use_exposure_time=True):
    """
    Return True if DICOM dataset is CT, False otherwise (e.g., CBCT)

    Args:
        dcm (pydicom.Dataset) DICOM object to test
        use_exposure_time (bool) if True use Exposure Time, else use Manufacturer
    """
    try:
        if dcm.get((0x0008, 0x0060)).value == "CT":
            if use_exposure_time:
                try:
                    exposure_t = dcm.get((0x0018, 0x1150)).value
                except (AttributeError, TypeError):
                    xray_curr = dcm.get((0x0018, 0x1151)).value
                    exposure = dcm.get((0x0018, 0x1152)).value
                    exposure_t = (1000*exposure) / xray_curr
                
                return exposure_t > 200
            elif dcm.get((0x0008, 0x0070)).value == "ELEKTA":
                return False
            else:
                return True
        else:
            return False
    except (PermissionError, AttributeError, TypeError):
        return False


def get_directory_level(path1, path2):
    """
    Return the number of levels that separate each path with the common parent path
    """
    commonpath = os.path.commonpath([path1, path2])
    relpath1 = os.path.relpath(path1, start=commonpath)
    relpath2 = os.path.relpath(path2, start=commonpath)
    return len(pathlib.Path(relpath1).parents), len(pathlib.Path(relpath2).parents)


def load_folder(path):
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
                if is_CT(dcm, use_exposure_time=False):
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
            data.extend(load_folder(folder))

    return data


def load_patient(path, log=None):
    """
    Load a patient folder and return images with dose and rtplan

    Args:
        path (str) path to patient folder
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

    id = int(pathlib.Path(path).name)
    
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
            if max(get_directory_level(rtdose.path, ct.path)) > 1:
                continue

            if is_CT_RTDOSE_match(ct, rtdose):
                ct.add_rtdose(rtdose, log)
                done = True
                break

        if not(done) and not(log is None):
            log.warning(f"WARNING: RTDOSE at {rtdose.path} found no matching CT")

    # group CT with struct
    for rtstruct in filter(lambda i: isinstance(i, dicom_class.RTSTRUCT), patient_data):
        done = False
        for ct in filter(lambda i: isinstance(i, dicom_class.CT), patient_data):
            if is_CT_RTSTRUCT_match(ct, rtstruct):
                ct.add_rtstruct(rtstruct, log)
                done = True
                break

        if not(done) and not(log is None):
            log.warning(f"WARNING: RTSTRUCT at {rtstruct.path} found no matching CT")
    
    return dicom_class.Patient(
        id_=id,
        ct=list(filter(lambda i: isinstance(i, dicom_class.CT), patient_data)),
        cbct=list(filter(lambda i: isinstance(i, dicom_class.CBCT), patient_data)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to dataset patients folder')
    parser.add_argument('--output', type=str, required=True, help='path to pickle file to save')
    parser.add_argument('--save_freq', type=int, default=1, help='frequency (no of patients) to save')
    args = parser.parse_args()

    data = []
    all_patients_to_load = glob.glob(os.path.join(args.input, "*"))
    for i, p in tqdm.tqdm(enumerate(all_patients_to_load), total=len(all_patients_to_load), ncols=50):
        if not(os.path.isdir(p)):
            continue

        p = load_patient(path=p, log=f"./{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
        
        # set base path
        p.update_study_base_path(args.input)

        data.append(p)
        
        if ((i+1) % args.save_freq) == 0:
            print("saving in pkl..")
            with open(args.output, "wb") as f:
                pickle.dump(data, f)

    print(f"\n {len(data)} patients loaded")

    print("saving in pkl..")
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    print("Done")
