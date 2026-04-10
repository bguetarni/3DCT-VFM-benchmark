import tqdm
import os
import argparse
import pickle
import torch
import time
from difflib import SequenceMatcher
from radiomics import featureextractor
import dicom2nifti
import cv2 as cv

from models import CTFM, SuPreM, VISTA3D, CT_CLIP
from datasets import cohorts_map


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to folder containing cohorts PICKLE files')
    parser.add_argument('--output', type=str, required=True, help='path to folder to save features')
    parser.add_argument('--nifti_tmp_folder', type=str, default="./tmp", help='path to folder to save Nifti files')
    parser.add_argument('--overwrite', action="store_true", default=False, help='whether to overwrite features file if already existing')
    parser.add_argument('--type', type=str, required=True, choices=["radiomics", "ct-fm", "suprem", "vista3d", "ct-clip"], help="type of features")
    parser.add_argument('--cohort', type=str, required=True, choices=list(cohorts_map.keys()), 
                        help='which cohort to build (change for certain parts)')
    parser.add_argument('--gpu', type=str, default="", help='GPU to use')
    args = parser.parse_args()

    # disable dicom2nifti slice increment validation to avoid errors
    # see https://github.com/icometrix/dicom2nifti/issues/36
    dicom2nifti.settings.disable_validate_slice_increment()

    # build output path and check if file already exists
    out_path = os.path.join(args.output, args.cohort, f"{args.type}.pkl")
    os.makedirs(os.path.split(out_path)[0], exist_ok=True)
    if os.path.exists(out_path) and not(args.overwrite):
        print(f"WARNING: exiting because destination file already exists ({out_path}). To overwrite, set argument --overwrite to True")
        exit(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device ", device)

    input_path = os.path.join(args.input, f"{args.cohort}.pickle")
    with open(input_path, "rb") as f:
        print("loading patients...")
        patients = pickle.load(f)
        print(f"found {len(patients)} patients in {input_path}")

    print(f"loading model {args.type}...")
    match args.type:
        case "ct-fm":
            infer, preprocess, model = CTFM.load(device)
        case "suprem":
            infer, preprocess, model = SuPreM.load(device)
        case "vista3d":
            infer, preprocess, model = VISTA3D.load(device)
        case "ct-clip":
            infer, preprocess, model = CT_CLIP.load(device)
        case "radiomics":
            params_file = "./radiomics.yaml"
            # script may run in parallel
            # save nifti files in a temporary folder with unique timestamp to avoid conflicts
            # to ensure different runs of script do not overwrite each other's nifti files, add PID
            dtime = time.strftime("%Y%m%d%H%M%S", time.gmtime())
            ct_nifti_path = os.path.join(args.nifti_tmp_folder, f"volume_{dtime}_{os.getpid()}.nii.gz")
            rtstruct_nifti_path = os.path.join(args.nifti_tmp_folder, f"gtv_{dtime}_{os.getpid()}.nii.gz")
        case _:
            raise ValueError(f"argument --type value not implemented: {args.type}")

    features = {}
    for id_, p in tqdm.tqdm(list(patients.items()), ncols=50):
        try:
            p.sort_imaging()   # sort images to recover first CT scan (i.e., CT0)
            input_path = p.ct[0].path
            bbox = p.ct[0].get_GTV_bbox()
        except (ValueError, KeyError, IndexError):
            continue
        
        if args.type == "radiomics":
            ct = p.ct[0]
            rtstruct = ct.rtstruct
            if rtstruct is None:
                print(f"patient {id_} has no RTSTRUCT, cannot compute radiomics features, skipping...")
                continue

            if ct.get_dcm_path().endswith(".nii.gz"):   # HECKTOR
                ct_nifti_path = ct.get_dcm_path()
                rtstruct_nifti_path = rtstruct.get_dcm_path()
            else: # convert DICOM roi to nifti format
                try:
                    # list all contours available in RTSTRUCT
                    # calculate similarity between contours name and 'gtv'
                    # select contour with name that most resemble 'gtv'
                    rois = rtstruct.get_all_OARs()
                    pairs =  [(i, similar("gtv", i.lower().replace(" ", ""))) for i in rois if "gtv" in i.lower()]
                    roi_name, _ = sorted(pairs, reverse=True, key=lambda i: i[1])[0]
                except IndexError:
                    try:   # check for PTV if GTV not available
                        rois = rtstruct.get_all_OARs()
                        pairs =  [(i, similar("gtv", i.lower().replace(" ", ""))) for i in rois if "ptv" in i.lower()]
                        roi_name, _ = sorted(pairs, reverse=True, key=lambda i: i[1])[0]
                    except IndexError:
                        continue
                
                # convert to nifti in temporary folder
                try:
                    ct.convert2nifti(ct_nifti_path)
                except dicom2nifti.exceptions.ConversionError:
                    print(f"ConversionError occured for patient {id_} CT, skipping...")
                    continue
                try:
                    rtstruct.convert2nifti(rtstruct_nifti_path, roi_name)
                except (cv.error, Exception):
                    continue

            try:
                # extract radiomics features
                extractor = featureextractor.RadiomicsFeatureExtractor()
                extractor.loadParams(params_file)
                featureVector = extractor.execute(ct_nifti_path, rtstruct_nifti_path, label=1, label_channel=0)
            except ValueError as e:
                print(f"ValueError occured for patient {id_}: {e}")
                continue
            
            features[id_] = {}
            for k, v in featureVector.items():
                if k.split("_")[0] == "diagnostics":
                    continue
                features[id_][k] = v
        else:
            try:
                output = infer(input_path, bbox, preprocess, model, device)
                features[id_] = output.flatten()
            except (Exception, RuntimeError) as e:
                print("Error/Exception occured: ", e)
                continue

    # save to pickle file
    with open(out_path, "wb") as f:
        pickle.dump(features, f)
    print("Done.")
