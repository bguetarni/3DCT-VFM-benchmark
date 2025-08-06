import tqdm, os, argparse, pickle
import pandas
import numpy as np

from radiomics import featureextractor
from rt_utils import RTStructBuilder
import SimpleITK as sitk
from dicompylercore import dvhcalc
import torch
from lighter_zoo import SegResEncoder
from monai.transforms import Compose, LoadImage, EnsureType, Orientation, ScaleIntensityRange, CropForeground
from totalsegmentator.python_api import totalsegmentator

TOTAL_SEG_CLASSES = {
    "head_glands_cavities": {7: "parotid_gland_left", 8: "parotid_gland_right", 9: "submandibular_gland_right",  10: "submandibular_gland_left"},
    "headneck_muscles": {3: "superior_pharyngeal_constrictor", 4: "middle_pharyngeal_constrictor", 5: "inferior_pharyngeal_constrictor"},
    "craniofacial_structures": {1: "mandible"},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to the cohort PICKLE file')
    parser.add_argument('--output', type=str, required=True, help='path to folder to save results')
    parser.add_argument('--tmp_folder', type=str, required=True, help='path where temporary files (Nifti, DICOM) are saved')
    
    # features to compute
    parser.add_argument('--radiomics', type=str, default=None, help="path to the pyradiomics params file, if None then not applied")
    parser.add_argument('--dosiomics', type=str, default=None, help="path to the pyradiomics params file, if None then not applied")
    parser.add_argument('--dvh', action="store_true", default=False)
    parser.add_argument('--deepNN', action="store_true", default=False)
    
    # additional parameters
    parser.add_argument('--rx_dose', type=int, default=70, help="prescribed value for RT, used to normalise DVH")
    parser.add_argument('--apply_total_seg', action="store_true", default=True, help="to apply TotalSegmentator if segmentations needed")    
    parser.add_argument('--gpu', type=str, default="", help='GPU to use')
    args = parser.parse_args()

    # TODO: add arguments:
    #   dcm2niix conversion argument
    #   other pyradiomics image types (LoG, ...)
    #   slides artifacts removing

    if args.apply_total_seg or args.deepNN:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.input, "rb") as f:
        patients = pickle.load(f)

    print(f"found {len(patients)} patients in {args.input_path}")

    os.makedirs(args.tmp_folder, exist_ok=True)

    for p in tqdm.tqdm(patients, ncols=50):
        p.sort_imaging()
        ct = p.ct[0]
        dose = ct.rtdose

        out_path = os.path.join(args.output, str(p.id))
        os.makedirs(out_path, exist_ok=True)

        if not(args.radiomics is None):
            print("computing radiomics...")
            extractor = featureextractor.RadiomicsFeatureExtractor()
            extractor.loadParams(args.radiomics)
            
            features = []
            
            if args.apply_total_seg:
                nii_path = os.path.join(args.tmp_folder, "volume.nii.gz")

                for task, oars in TOTAL_SEG_CLASSES.items():
                    mask_path = os.path.join(args.tmp_folder, f"{task}.nii.gz")
                    ct.apply_totalsegmentator(task=task, tmp_nii_input=nii_path, tmp_nii_output=mask_path, overwrite_nifti=True)

                    for oar_id, oar_name in oars.items():
                        featureVector = extractor.execute(nii_path, mask_path, label=oar_id, label_channel=0)
                        for fname, fvalue in featureVector.items():
                            type_, class_, name_ = fname.split("_")
                            features.append({"oar": oar_name, "type": type_, "class": class_, "name": name_, "value": fvalue})
            else:
                raise NotImplementedError #TODO implement using rtstruct
            
            # save to csv
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "radiomics.csv"))

        if not(args.dosiomics is None) and not(dose is None):
            print("computing dosiomics...")
            nii_path = os.path.join(args.tmp_folder, "dose.nii.gz")
            dose.convert2nifti(nii_path)

            extractor = featureextractor.RadiomicsFeatureExtractor()
            extractor.loadParams(args.dosiomics)
            
            features = []
            
            if args.apply_total_seg:
                for task, oars in TOTAL_SEG_CLASSES.items():
                    mask_path = os.path.join(args.tmp_folder, f"{task}.nii.gz")

                    # create TotalSegmentator mask
                    if args.radiomics is None:
                        ct_nii_path = os.path.join(args.tmp_folder, "volume.nii.gz")
                        ct.convert2nifti(ct_nii_path)
                        ct.apply_totalsegmentator(task=task, tmp_nii_input=ct_nii_path, tmp_nii_output=mask_path)

                    for oar_id, oar_name in oars.items():
                        featureVector = extractor.execute(nii_path, mask_path, label=oar_id, label_channel=0)
                        for fname, fvalue in featureVector.items():
                            type_, class_, name_ = fname.split("_")
                            features.append({"oar": oar_name, "type": type_, "class": class_, "name": name_, "value": fvalue})
            else:
                raise NotImplementedError #TODO implement using rtstruct
            
            # save to csv
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "dosiomics.csv"))

        if args.dvh:
            print("computing DVH features...")

            # Create new RT Struct. 
            # Requires the DICOM series path for the RT Struct.
            rtstruct = RTStructBuilder.create_new(dicom_series_path=ct.path)

            features = []
            
            if args.apply_total_seg:
                for task, oars in TOTAL_SEG_CLASSES.items():
                    mask_path = os.path.join(args.tmp_folder, f"{task}.nii.gz")

                    # create TotalSegmentator mask
                    if args.radiomics is None:
                        ct_nii_path = os.path.join(args.tmp_folder, "volume.nii.gz")
                        ct.convert2nifti(ct_nii_path)
                        ct.apply_totalsegmentator(task=task, tmp_nii_input=ct_nii_path, tmp_nii_output=mask_path)
                    
                    # read mask
                    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                    
                    for oar_id, oar_name in oars.items():
                        oar_mask = mask.copy()
                        oar_mask[oar_mask != oar_id] = 0
                        oar_mask[oar_mask == oar_id] = 1

                        oar_mask = np.moveaxis(oar_mask, 0, -1)
                        oar_mask = np.flip(oar_mask, axis=0).astype("bool")
                        
                        rtstruct.add_roi(mask=oar_mask, name=oar_name)
                        rtstruct.save(os.path.join(args.tmp_folder, "mask.dcm"))

                        calcdvh = dvhcalc.get_dvh(structure=os.path.join(args.tmp_folder, "mask.dcm"), 
                                                  dose=dose.get_dcm_path(), 
                                                  roi={s.ROIName: s.ROINumber for s in rtstruct.ds.StructureSetROISequence}[oar_name])
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
            else:
                raise NotImplementedError #TODO implement using rtstruct
            
            # save to csv
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "dvh.csv"))

        if args.deepNN:
            nii_path = os.path.join(args.tmp_folder, "volume.nii.gz")
            ct.convert2nifti(nii_path)

            if args.apply_total_seg:
                for task, oars in TOTAL_SEG_CLASSES.items():
                    # create mask
                    mask = totalsegmentator(nii_path, task=task)
                    mask = np.moveaxis(mask.get_fdata(), -1, 0)

                    for oar_id, oar_name in oars.items():
                        mask[mask != oar_id] = 0
                        mask[mask == oar_id] = 1

                        img = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
                        img[np.invert(mask.astype("bool"))] = -1024
                        img = sitk.GetImageFromArray(img)
                        sitk.WriteImage(img, nii_path)

                        # Load pre-trained model
                        model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
                        model.eval().to(device=device)

                        # Preprocessing pipeline
                        preprocess = Compose([
                            LoadImage(ensure_channel_first=True),  # Load image and ensure channel dimension
                            EnsureType(),                         # Ensure correct data type
                            Orientation(axcodes="SPL"),           # Standardize orientation
                            
                            # Scale intensity to [0,1] range, clipping outliers
                            ScaleIntensityRange(
                                a_min=-1024,    # Min HU value
                                a_max=2048,     # Max HU value
                                b_min=0,        # Target min
                                b_max=1,        # Target max
                                clip=True       # Clip values outside range
                            ),
                            CropForeground()    # Remove background to reduce computation
                        ])

                        # Preprocess input
                        input_tensor = preprocess(nii_path)

                        # Run inference
                        with torch.no_grad():
                            output = model(input_tensor.unsqueeze(0).to(device=device))[-1]

                            # Average pooling compressed the feature vector across all patches. If this is not desired, remove this line and 
                            # use the output tensor directly which will give you the feature maps in a low-dimensional space.
                            avg_output = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()
                        
                        features = avg_output.cpu().numpy()
                        features = {i: j for i,j in enumerate(features.flatten())}
            else:
                raise NotImplementedError #TODO implement using rtstruct
            
            # save to csv
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "deepnn.csv"))
