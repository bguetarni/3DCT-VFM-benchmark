import tqdm
import os
import argparse
import pickle
import pandas
import numpy as np

from radiomics import featureextractor
from rt_utils import RTStructBuilder
import SimpleITK as sitk
from dicompylercore import dvhcalc
import torch
from monai import transforms as monai_transforms
from lighter_zoo import SegResEncoder

import dicom_class


def create_mask_original(dicom_obj, mask_path):
    """
    dicom_obj (CT,RTDOSE) dicom object for copying image metadata and computing mask optionally
    mask_path (str) path to save mask in Nifti format
    """

    if isinstance(dicom_obj, dicom_class.CT):
        oars = dicom_obj.rtstruct.get_all_OARs()
    elif isinstance(dicom_obj, dicom_class.RTDOSE):
        oars = dicom_obj.get_parent().rtstruct.get_all_OARs()
    else:
        return None
    
    oars_mask = dict()
    mask = None
    i = 1
    for oar in oars:
        try:
            if isinstance(dicom_obj, dicom_class.CT):
                roi_mask = dicom_obj.rtstruct.get_structure_mask(oar)
            else:
                roi_mask = dicom_obj.get_structure_mask(oar)
    
            if mask is None:
                mask = np.zeros_like(roi_mask, dtype=np.int16)
            
            mask[roi_mask] = i
            oars_mask.update({oar: i})
            i += 1
        except Exception:
            pass
    
    # copy image metadata into mask
    mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask.CopyInformation(dicom_obj.get_sitk_image())
    
    # save mask
    sitk.WriteImage(mask, mask_path)
    
    return oars_mask


def compute_radiomics_generic(nii_path, mask_path, oars, radiomics_yaml):
    """
    Compute radiomics using OAR segmentation

    args
        nii_path (str) path to image nifti data
        mask_path (str) path to mask nifti data
        oars (dict) dictionnary containing mapping between RTSTRUCT OAR names and standard ones
        radiomics_yaml (str) path to the pyradiomics params file
    """

    # create radiomics feature extractor with parameters from file
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.loadParams(radiomics_yaml)

    # compute radiomics
    features = []
    for name, id_ in oars.items():
        try:
            featureVector = extractor.execute(nii_path, mask_path, label=id_, label_channel=0)
            for fname, fvalue in featureVector.items():
                type_, class_, name_ = fname.split("_")
                if class_ == "diagnostics":
                    continue
                features.append({"oar": name, "type": type_, "class": class_, "name": name_, "value": fvalue})
        except ValueError:
            continue
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to the cohort PICKLE file')
    parser.add_argument('--output', type=str, required=True, help='path to folder to save results')
    parser.add_argument('--tmp_folder', type=str, default="./", help='path where temporary files (Nifti, DICOM) are saved')
    
    # features to compute
    parser.add_argument('--radiomics', type=str, default=None, help="path to the pyradiomics params file, if None then not applied")
    parser.add_argument('--dosiomics', type=str, default=None, help="path to the pyradiomics params file, if None then not applied")
    parser.add_argument('--dvh', action="store_true", default=False)
    parser.add_argument('--deepNN', type=str, default=False, choices=["ct-fm"], help="name of foundation model to use")
    
    # additional parameters
    parser.add_argument('--rx_dose', type=int, default=70, help="prescribed value for RT, used to normalise DVH")
    parser.add_argument('--gpu', type=str, default="", help='GPU to use (e.g., 0)')
    args = parser.parse_args()

    if args.deepNN:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device ", device)

    with open(args.input, "rb") as f:
        print("loading patients...")
        patients = pickle.load(f)
        print(f"found {len(patients)} patients in {args.input}")

    # create temporary folder for saving nifti files
    os.makedirs(args.tmp_folder, exist_ok=True)

    CT_NII_PATH = os.path.join(args.tmp_folder, "volume.nii.gz")
    DOSE_NII_PATH = os.path.join(args.tmp_folder, "dose.nii.gz")
    CT_MASK_PATH = os.path.join(args.tmp_folder, "volume_mask.nii.gz")
    DOSE_MASK_PATH = os.path.join(args.tmp_folder, "dose_mask.nii.gz")

    # remove temporary volume nifti file
    if os.path.exists(CT_NII_PATH):
        os.remove(CT_NII_PATH)

    # remove temporary dose nifti file
    if os.path.exists(DOSE_NII_PATH):
        os.remove(DOSE_NII_PATH)

    for p in tqdm.tqdm(patients, ncols=50):
        # sort images to recover first CT scans (i.e., CT0)
        p.sort_imaging()

        # select first scan
        ct = p.ct[0]

        # check that CT image has an RTDOSE
        if ct.rtdose is None:
            print(f"WARNING: patient {p.id} CT0 does not have an RTDOSE, skipping..")
            continue

        # define and delete if existing the patient output directory
        out_path = os.path.join(args.output, str(p.id))
        if os.path.isdir(out_path):
            try:
                os.rmdir(out_path)
            except (FileNotFoundError, OSError):
                pass
        
        # create the output directory of patient
        os.makedirs(out_path, exist_ok=True)

        # convert from DICOM to Nifti
        ct.convert2nifti(CT_NII_PATH)
        ct.rtdose.convert2nifti(DOSE_NII_PATH)

        # create OARs mask get oar name id mapping
        print("creating OARs masks based on original contours...")
        oars = create_mask_original(ct, CT_MASK_PATH, args.oar_names, int(p.id))
        create_mask_original(ct.rtdose, DOSE_MASK_PATH, args.oar_names, int(p.id))
        
        if args.radiomics:
            print("computing radiomics...")
            features = compute_radiomics_generic(CT_NII_PATH, CT_MASK_PATH, oars, args.radiomics)
            print("saving radiomics in csv...")
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "radiomics.csv"))

        if args.dosiomics and ct.rtdose:
            print("computing dosiomics...")
            features = compute_radiomics_generic(DOSE_NII_PATH, DOSE_MASK_PATH, oars, args.dosiomics)
            print("saving dosiomics in csv...")
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "dosiomics.csv"))

        if args.dvh and ct.rtdose:
            print("computing DVH features...")

            # Create new RT Struct.
            # Requires the DICOM series path for the RT Struct.
            rtstruct = RTStructBuilder.create_new(dicom_series_path=ct.path)
                    
            # read mask
            mask = sitk.GetArrayFromImage(sitk.ReadImage(CT_MASK_PATH))
            
            features = []
            for oar_name, oar_id in oars.items():
                # create oar mask
                oar_mask = mask.copy()
                oar_mask[mask != oar_id] = 0
                oar_mask[mask == oar_id] = 1
                oar_mask = np.moveaxis(oar_mask, 0, -1)
                oar_mask = np.flip(oar_mask, axis=0).astype("bool")
                
                # add oar to rtstruct object and save to DICOM file
                rtstruct.add_roi(mask=oar_mask, name=oar_name)
                rtstruct.save(os.path.join(args.tmp_folder, "mask.dcm"))

                # compute dvh features
                roi_id = {roi.ROIName: roi.ROINumber for roi in rtstruct.ds.StructureSetROISequence}[oar_name]
                calcdvh = dvhcalc.get_dvh(structure=os.path.join(args.tmp_folder, "mask.dcm"), 
                                          dose=ct.rtdose.get_dcm_path(), 
                                          roi=roi_id)
                calcdvh.rx_dose = args.rx_dose
                calcdvh = calcdvh.relative_volume # transform into relative volume DVH
                calcdvh = calcdvh.absolute_dose() # transform to absolute dose
                dvh = {
                    "mean": calcdvh.mean,
                    "min": calcdvh.min,
                    "max": calcdvh.max,
                    "D98": calcdvh.dose_constraint(98, volume_units='%'),
                    **{f"D{v}": calcdvh.dose_constraint(v, volume_units='%') for v in range(10,100,10)},
                    **{f"V{d}": calcdvh.volume_constraint(d, dose_units='Gy') for d in range(5,80,5)},
                }

                for k, v in dvh.items():
                    features.append({"oar": oar_name, "name": k, "value": v})

            # save to csv
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "dvh.csv"))

        if args.deepNN:
            print("computing deep nn features...")            

            # read mask
            mask = sitk.GetArrayFromImage(sitk.ReadImage(CT_MASK_PATH))

            features = []
            for oar_name, oar_id in oars.items():
                TMP_OAR_NII = os.path.join(args.tmp_folder, f"{oar_name}.nii.gz")

                oar_mask = mask.copy()
                oar_mask[mask != oar_id] = 0
                oar_mask[mask == oar_id] = 1

                img = sitk.GetArrayFromImage(sitk.ReadImage(CT_NII_PATH))
                img[np.invert(oar_mask.astype("bool"))] = -1024
                sitk.WriteImage(sitk.GetImageFromArray(img), TMP_OAR_NII)

                # Load pre-trained model
                if args.deepNN == "ct-fm":
                    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")

                    # Preprocessing pipeline, do not change it !
                    preprocess = monai_transforms.Compose(
                        [
                            monai_transforms.LoadImage(ensure_channel_first=True),
                            monai_transforms.EnsureType(),
                            monai_transforms.Orientation(axcodes="SPL"),
                            monai_transforms.ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
                            monai_transforms.CropForeground(margin=50)
                        ]
                    )
                else:
                    raise NotImplementedError(f"foundation model {args.deepNN} not implemented")
                
                # send model to device
                model.eval().to(device=device)

                # Preprocess input
                input_tensor = preprocess(TMP_OAR_NII)

                # Run inference
                with torch.no_grad():
                    try:
                        output = model(input_tensor.unsqueeze(0).to(device=device))
                    except RuntimeError: # might throw an error if the ROI is too small
                        continue

                # Average pooling compressed the feature vector across all patches.
                if args.deepNN == "ct-fm":
                    output = torch.nn.functional.adaptive_avg_pool3d(output[-1], 1)
                
                # convert features to numpy and add to features list
                fts = output.squeeze().cpu().numpy().flatten()
                for i, f in enumerate(fts):
                    features.append({"oar": oar_name, "name": i, "value": f})
            
            # save to csv
            pandas.DataFrame(features).to_csv(os.path.join(out_path, f"deepnn({args.deepNN}).csv"))

    print("Done.")
