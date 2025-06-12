import logging
import os, glob, pathlib
import pydicom
import pandas

import dicom_class
import dicom_utils

# (0008, 0060) Modality
# 'CT', 'RTSTRUCT', 'RTDOSE'
# CT vs CBCT: (0008, 0070) Manufacturer of CBCT is 'ELEKTA'
# threshold for xerostomia is SSF<500 mg/min

def generic_parsing(df, type, visitID_key, value_key=None, filter_pairs=None, sample_keys=None, sample_is_column=True):
    """
    Generic parsing of clinical CSV

    Args:
        df (pandas.DataFrame) CSV to parse
        type (str) the name of type of measurments [SSF, DOSIMETRY, ...]
        visitID_key (str) name of the column of visit ID
        value_key (str) name of the column of the measurement value
        filter_pairs (str) list of pairs of column names and values to filter the table (each column can have multiple values)
        sample_keys (List[str], str) names of columns that  are the sample or of the column that contain the sample name
        sample_is_column (bool) if True consider that sample_keys is name of columns that contain the value, 
                else name of the columns that contain the sample names
    """

    if sample_is_column and not(sample_keys is None):
        assert isinstance(sample_keys, list), "if 'sample_is_column' is True then 'sample_keys' must be a list of column names"

    if not(sample_is_column) and not(sample_keys is None):
        assert isinstance(sample_keys, str), "if 'sample_is_column' is False, then 'sample_keys' must be a string as the name of the column containing the sample names"
        assert not(value_key is None), "if 'sample_is_column' is False, then 'value_key' must be provided"

    if not filter_pairs is None:
        for k, v in filter_pairs:
            if not isinstance(v, list):
                v = [v]

            # filter the table
            df = df[df[k].isin(v)]
    
    # for each row populate clinical measurements data
    result = []
    for _, row in df.iterrows():
        if sample_keys is None:
            result.append({"type": type, "value": row[value_key], "visitID": row[visitID_key]})
        else:
            # for each sample add data and name
            if sample_is_column:
                # sample_keys are the name of the samples columns
                for k in sample_keys:
                    result.append({"type": type, "value": row[k], "visitID": row[visitID_key], "sample": k})
            else:
                # sample_keys is the column containing samples names
                result.append({"type": type, "value": row[value_key], "visitID": row[visitID_key], "sample": row[sample_keys]})
    return result

def salivation_parsing(df):
    return generic_parsing(df, 
                           filter_pairs=[("MEASTYP", ["Stimulated salivation flow"])],
                           type="SSF",
                           visitID_key="VISITID",
                           value_key="MEAS_VAL")

def dosimetry_parsing(df):
    return generic_parsing(df, 
                           type="DOSIMETRY",
                           visitID_key="DOSISEQ",
                           sample_keys=["PAROH_DOSE", "PAROC_DOSE", "SMAXH_DOSE", "SMAXC_DOSE", "MOUTH_DOSE"],
                           sample_is_column=True)

def mda_parsing(df):
    return generic_parsing(df, 
                           type="MDA",
                           visitID_key="VISITID",
                           sample_keys=['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9',
                                        'Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19',
                                        'Q20','Q21','Q22','Q23','Q24','Q25','Q26','Q27','Q28'],
                           sample_is_column=True)

def aetox_parsing(df):
    return generic_parsing(df,
                           filter_pairs=[("AETERM", ["DYSPHAGIE", "XEROSTOMIE"])],
                           type="AE",
                           visitID_key="EPOC",
                           value_key="AEGRMX",
                           sample_keys="AETERM",
                           sample_is_column=False)

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

def load_patient(path, 
                 id_map, 
                 PATIENT_DESCRIPTION_csv=None,
                 EFFICACY_csv=None,
                 SALIVATION_DATA_csv=None,
                 TREATMENT_csv=None,
                 DOSIMETRY_csv=None,
                 MDA_csv=None,
                 AETOXGEN_csv=None,
                 log=None):
    """
    Load a patient folder and return images with dose and rtplan

    Args:
        path (str) path to patient folder
        id_map (str) path to CSV mapping folder IDs to clinical data IDs
        PATIENT_DESCRIPTION_csv (str) path to PATIENT_DESCRIPTION csv
        EFFICACY_csv (str) path EFFICACY to csv
        SALIVATION_DATA_csv (str) path SALIVATION_DATA to csv
        TREATMENT_csv (str) path TREATMENT to csv
        DOSIMETRY_csv (str) path DOSIMETRY to csv
        MDA_csv (str) path MDA to csv
        AETOXGEN_csv (str) path AETOXGEN to csv
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
    clinical = None
    for sub_df in (PATIENT_DESCRIPTION_csv, EFFICACY_csv, TREATMENT_csv):
        if sub_df is None:
            continue

        sub_df = pandas.read_csv(sub_df, sep=";", encoding='ISO-8859-1')
        
        # convert USUBJID and filter data for patient id only
        # check patient is in CSV data
        sub_df = sub_df[sub_df["USUBJID"].astype(int) == id]
        if len(sub_df) == 0:
            if not log is None:
                log.warning(f"WARNING: patient folder id {id} not found in clinical data")
            else:
                print(f"WARNING: patient folder id {id} not found in clinical data")
            continue

        if clinical is None:
            clinical = sub_df
        else:
            # merge DataFrames based on common columns
            common_columns = set(clinical.columns).intersection(set(sub_df.columns))
            clinical = pandas.merge(clinical, sub_df, on=list(common_columns), how='inner')

    # load clinical measurements
    clinical_measurements = []
    for sub_df, fn in [
        (SALIVATION_DATA_csv, salivation_parsing),
        (DOSIMETRY_csv, dosimetry_parsing),
        (MDA_csv, mda_parsing),
        (AETOXGEN_csv, aetox_parsing),
    ]:
        if sub_df is None:
            continue

        sub_df = pandas.read_csv(sub_df, sep=";", encoding='ISO-8859-1')

        # convert USUBJID and filter data for patient id only
        # check patient is in CSV data
        sub_df = sub_df[sub_df["USUBJID"].astype(int) == id]
        if len(sub_df) == 0:
            if not log is None:
                log.warning(f"WARNING: patient folder id {id} not found in clinical data")
            else:
                print(f"WARNING: patient folder id {id} not found in clinical data")
            continue

        clinical_measurements.extend(fn(sub_df))

    
    return dicom_class.Patient(
        id_=id,
        ct=list(filter(lambda i: isinstance(i, dicom_class.CT), patient_data)),
        cbct=list(filter(lambda i: isinstance(i, dicom_class.CBCT), patient_data)),
        clinical=clinical.to_dict('records')[0],
        clinical_measurements=clinical_measurements,
    )
