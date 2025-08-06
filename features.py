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
        print("using device ", device)

    with open(args.input, "rb") as f:
        print("loading patients...")
        patients = pickle.load(f)

    print(f"found {len(patients)} patients in {args.input}")

    os.makedirs(args.tmp_folder, exist_ok=True)

    for p in tqdm.tqdm(patients, ncols=50):
        p.sort_imaging()
        ct = p.ct[0]

        out_path = os.path.join(args.output, str(p.id))
        os.makedirs(out_path, exist_ok=True)

        CT_NII_PATH = os.path.join(args.tmp_folder, "volume.nii.gz")
        DOSE_NII_PATH = os.path.join(args.tmp_folder, "dose.nii.gz")

        if os.path.exists(CT_NII_PATH):
            os.remove(CT_NII_PATH)

        if os.path.exists(DOSE_NII_PATH):
            os.remove(DOSE_NII_PATH)

        # convert from DICOM to Nifti
        ct.convert2nifti(CT_NII_PATH)

        if args.apply_total_seg and (args.radiomics or args.dosiomics or args.dvh or args.deepNN):
            print("computing segmentations first...")

            for task in TOTAL_SEG_CLASSES.keys():
                ct.apply_totalsegmentator(task=task, 
                                          tmp_nii_input=CT_NII_PATH, 
                                          tmp_nii_output=os.path.join(args.tmp_folder, f"{task}.nii.gz"))

        if args.radiomics:
            print("computing radiomics...")
            extractor = featureextractor.RadiomicsFeatureExtractor()
            extractor.loadParams(args.radiomics)

            features = []
            if args.apply_total_seg:
                for task, oars in TOTAL_SEG_CLASSES.items():
                    mask_path = os.path.join(args.tmp_folder, f"{task}.nii.gz")

                    for oar_id, oar_name in oars.items():
                        featureVector = extractor.execute(CT_NII_PATH, mask_path, label=oar_id, label_channel=0)
                        for fname, fvalue in featureVector.items():
                            type_, class_, name_ = fname.split("_")
                            features.append({"oar": oar_name, "type": type_, "class": class_, "name": name_, "value": fvalue})
            else:
                raise NotImplementedError #TODO implement using rtstruct
            
            # save to csv
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "radiomics.csv"))

        if args.dosiomics and ct.rtdose:
            print("computing dosiomics...")
            ct.rtdose.convert2nifti(DOSE_NII_PATH)

            extractor = featureextractor.RadiomicsFeatureExtractor()
            extractor.loadParams(args.dosiomics)
            
            features = []
            if args.apply_total_seg:
                for task, oars in TOTAL_SEG_CLASSES.items():
                    mask_path = os.path.join(args.tmp_folder, f"{task}.nii.gz")

                    for oar_id, oar_name in oars.items():
                        featureVector = extractor.execute(DOSE_NII_PATH, mask_path, label=oar_id, label_channel=0)
                        for fname, fvalue in featureVector.items():
                            type_, class_, name_ = fname.split("_")
                            features.append({"oar": oar_name, "type": type_, "class": class_, "name": name_, "value": fvalue})
            else:
                raise NotImplementedError #TODO implement using rtstruct
            
            # save to csv
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "dosiomics.csv"))

        if args.dvh and ct.rtdose:
            print("computing DVH features...")

            # Create new RT Struct. 
            # Requires the DICOM series path for the RT Struct.
            rtstruct = RTStructBuilder.create_new(dicom_series_path=ct.path)

            features = []
            if args.apply_total_seg:
                for task, oars in TOTAL_SEG_CLASSES.items():
                    mask_path = os.path.join(args.tmp_folder, f"{task}.nii.gz")
                    
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
                                                  dose=ct.rtdose.get_dcm_path(), 
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
            print("computing deep nn features...")
            if args.apply_total_seg:
                for task, oars in TOTAL_SEG_CLASSES.items():

                    # read mask
                    mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.tmp_folder, f"{task}.nii.gz")))

                    for oar_id, oar_name in oars.items():

                        TMP_OAR_NII = os.path.join(args.tmp_folder, f"{oar_id}.nii.gz")

                        oar_mask = mask.copy()
                        oar_mask[oar_mask != oar_id] = 0
                        oar_mask[oar_mask == oar_id] = 1

                        img = sitk.GetArrayFromImage(sitk.ReadImage(CT_NII_PATH))
                        img[np.invert(oar_mask.astype("bool"))] = -1024
                        sitk.WriteImage(sitk.GetImageFromArray(img), TMP_OAR_NII)

                        # Load pre-trained model
                        model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
                        model.eval().to(device=device)

                        # Preprocessing pipeline, do not change it !
                        preprocess = Compose([
                            LoadImage(ensure_channel_first=True),
                            EnsureType(),
                            Orientation(axcodes="SPL"),
                            ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
                            CropForeground()
                        ])

                        # Preprocess input
                        input_tensor = preprocess(TMP_OAR_NII)

                        # Run inference
                        with torch.no_grad():
                            output = model(input_tensor.unsqueeze(0).to(device=device))[-1]

                            # Average pooling compressed the feature vector across all patches.
                            avg_output = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()
                        
                        # convert features to numpy and build dictionnary
                        features = avg_output.cpu().numpy().flatten().tolist()
                        features = [dict(enumerate(features))]
            else:
                raise NotImplementedError #TODO implement using rtstruct
            
            # save to csv
            pandas.DataFrame(features).to_csv(os.path.join(out_path, "deepnn.csv"))

    print("Done.")
