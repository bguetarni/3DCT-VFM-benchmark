import tqdm, os, argparse, pickle
import pandas
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import radon
from radiomics import featureextractor
import nibabel
from rt_utils import RTStructBuilder
import SimpleITK as sitk
from dicompylercore import dvhcalc
import torch
from monai import transforms as monai_transforms

import DAClassification

try:
    from lighter_zoo import SegResEncoder
except ModuleNotFoundError:
    print("WARNING: module pydicom version error, if SegResEncoder to import use another env")
    pass

# try:
#     from fmcib.models import fmcib_model
# except ModuleNotFoundError:
#     print("WARNING: module pydicom version error, if fmcib_model to import use another env")
#     pass


def create_mask_original(dicom_obj, mask_path, oars_map, p_id):
    """
    dicom_obj (CT,RTDOSE) dicom object for copying image metadata and computing mask optionally
    mask_path (str) path to save mask in Nifti format
    oars_map (str) path to csv for mapping original OAR names to standard ones
    p_id (str,int) patient ID
    """
    # transform mapping from original to standard names in dict 
    # with original names as keys and standard names as values
    oars_map = pandas.read_csv(oars_map)
    oars_map = oars_map[oars_map["patient_id"] == int(p_id)]
    oars_map = oars_map.set_index('original_name').to_dict()['renamed_name']

    oars = dict()
    i = 1
    mask = None
    for oar_original_name, oar_standard_name in oars_map.items():
        try:
            if hasattr(dicom_obj, "rtstruct"):
                roi_mask = dicom_obj.rtstruct.get_structure_mask(oar_original_name)
            else:
                roi_mask = dicom_obj.parent.rtstruct.get_structure_mask(oar_original_name)
            
            if mask is None:
                mask = np.zeros_like(roi_mask, dtype=np.int16)
            
            mask[roi_mask] = i
            oars.update({oar_standard_name: i})
            i += 1
        except Exception:
            pass
    
    # copy image metadata into mask
    mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask.CopyInformation(dicom_obj.get_sitk_image())
    
    # save mask
    sitk.WriteImage(mask, mask_path)
    
    return oars

def create_mask_totalsegmentator(dicom_obj, nii_path, mask_path):
    """
    dicom_obj (CT,RTDOSE) dicome object for copying image metadata and computing mask optionally
    nii_path (srt) path to image nifti data
    mask_path (str) path to save mask in Nifti format
    """

    TOTALSEG_CLASS = {
        "head_glands_cavities": {7: "parotid_gland_left", 8: "parotid_gland_right", 9: "submandibular_gland_right",  10: "submandibular_gland_left"},
        "headneck_muscles": {3: "superior_pharyngeal_constrictor", 4: "middle_pharyngeal_constrictor", 5: "inferior_pharyngeal_constrictor"},
        "craniofacial_structures": {1: "mandible"},
        }
    
    oars = dict()
    i = 1
    mask = None
    for task in TOTALSEG_CLASS.keys():
        dicom_obj.apply_totalsegmentator(task=task, tmp_nii_input=nii_path, tmp_nii_output=mask_path)
        nib_mask = nibabel.load(mask_path)
        output = nib_mask.get_fdata()
        for oar_id, oar_name in TOTALSEG_CLASS[task].items():
            if mask is None:
                mask = np.zeros_like(output, dtype=np.int16)

            mask[output == oar_id] = i
            oars.update({oar_name: i})
            i += 1

    # copy image metadata into mask and save it
    mask = nibabel.Nifti1Image(mask, nib_mask.affine, nib_mask.header)
    nibabel.save(mask, mask_path)
    
    return oars

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
                _, class_, _ = fname.split("_")
                if class_ == "diagnostics":
                    continue
                features.append({"oar": name, "name": fname, "value": fvalue})
        except ValueError:
            continue
    return features

def get_DA_slices_index(ct_path):
    """
    Find slices with dental artifacts and return the indices

    args:
        ct_path (str) path to CT in nifti format for detecting artifacts
    """

    z_min, z_max = 20, -2
    y_min, y_max = 0, 350
    x_min, x_max = 50, -50
    MIN, MAX = -1024, 3000

    torch.cuda.set_device(0)
    network = DAClassification.LoadNet("./", 'testCheckpoint.pth.tar')
    network.cuda()

    image = DAClassification.LoadImage(ct_path)
    predicted_label = DAClassification.GetPredictions(image.cuda(), network)
    if predicted_label.cpu().numpy().item():
        original_stack = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))

        # This removes unwanted common features
        stack = original_stack[z_min:z_max, y_min:y_max, x_min:x_max]

        # Normalize the image
        stack = stack.astype(float)
        stack = np.clip(stack, MIN, MAX)
        stack = (stack - MIN) / (MAX - MIN)

        # Loop through all images in patient's stack of scans
        intensities = []
        for image in tqdm.tqdm(stack) :
            if np.sum(image) < 1.0e-8 :
                intensities.append(0.0)
                continue   # If the image is entirely black, just go to next image

            # Remove the patient's body from the images
            try :
                otsu = threshold_otsu(image)                  # Compute Otsu threshold
                fill = binary_fill_holes(np.array(image > otsu, dtype=int))  # Fill holes
                gauss_fill = gaussian(fill, sigma=10)        # Add Gaussian  blur
                fill = np.array(gauss_fill < 0.01, dtype=int)    # Threshold again
                image = np.multiply(image, fill)            # Crop out body from raw image
            except ValueError:
                # If image is None, just go to next image
                # remove_body() may return None if the slice is identically some value
                intensities.append(0.0)
                continue

            # Threshold the new image
            image = np.array(image > 0.02, dtype=int)

            theta = np.linspace(0., 180., 180, endpoint=False)
            sinogram = radon(image, theta=theta, circle=True)
            mean = np.mean(sinogram[120:-120, 40:-40])
            intensities.append(mean)

        # Find the slices with artifacts
        indices, _ = find_peaks(intensities, height=5e-10)
        
        # Convert indices to list of ints
        indices = np.asarray(indices, dtype="int") + z_min
        return list(indices)

def remove_slice(slice_index, mask_path):
    original_mask = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(original_mask)
    for z in slice_index:
        mask[z] = 0
    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(original_mask)
    sitk.WriteImage(mask, mask_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='path to the cohort PICKLE file')
    parser.add_argument('--output', type=str, required=True, help='path to csv file to save features')
    parser.add_argument('--tmp_folder', type=str, required=True, help='path where temporary files (Nifti, DICOM) are saved')
    parser.add_argument('--overwrite', action="store_true", default=False, help='weither to overwrite features file if already existing')
    
    # features to compute
    parser.add_argument('--radiomics', type=str, default=None, help="path to the pyradiomics params file, if None then not applied")
    parser.add_argument('--dosiomics', type=str, default=None, help="path to the pyradiomics params file, if None then not applied")
    parser.add_argument('--dvh', action="store_true", default=False)
    parser.add_argument('--deepNN', type=str, default=False, choices=["ct-fm", "fmcib"], help="name of foundation model to use")
    
    # additional parameters
    parser.add_argument('--filterDA', action="store_true", default=False, help="weither to filter slices with dental artifcats (computationally intensive)")
    parser.add_argument('--oar_source', type=str, default="original", choices=["original", "totalsegmentator"], help="wether to use original OAR segmentations or apply TotalSegmentator")
    parser.add_argument('--oar_names', type=str, default=None, help="path to csv file containing mapping between RTSTRUCT OAR names and standard ones")
    parser.add_argument('--rx_dose', type=int, default=70, help="prescribed value for RT, used to normalise DVH")
    parser.add_argument('--gpu', type=str, default="", help='GPU to use')
    args = parser.parse_args()

    if os.path.exists(args.output) and not(args.overwrite):
        print(f"WARNING: exiting because destination file already exists ({args.output}). To overwrite, set argument --overwrite to True")
        exit(0)

    if args.oar_source == "totalsegmentator" or args.deepNN:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device ", device)

    if args.oar_source == "original" and not(args.oar_names):
        raise ValueError("If using original OAR source then --oar_names argument must be specified")

    with open(args.input, "rb") as f:
        print("loading patients...")
        patients = pickle.load(f)
        print(f"found {len(patients)} patients in {args.input}")

    # create temporary folder for saving nifti files
    os.makedirs(args.tmp_folder, exist_ok=True)

    CT_NII_PATH = os.path.join(args.tmp_folder, "volume.nii.gz")
    DOSE_NII_PATH = os.path.join(args.tmp_folder, "dose.nii.gz")

    # remove temporary volume nifti file
    if os.path.exists(CT_NII_PATH):
        os.remove(CT_NII_PATH)

    # remove temporary dose nifti file
    if os.path.exists(DOSE_NII_PATH):
        os.remove(DOSE_NII_PATH)

    features = []
    for p in tqdm.tqdm(patients, ncols=50):
        # sort images to recover first CT scans (i.e., CT0)
        p.sort_imaging()

        # select first scan
        ct = p.ct[0]

        # check that CT image has an RTDOSE
        if ct.rtdose is None:
            print(f"WARNING: patient {p.id} CT0 does not have an RTDOSE, skipping..")
            continue

        # convert from DICOM to Nifti
        ct.convert2nifti(CT_NII_PATH)
        ct.rtdose.convert2nifti(DOSE_NII_PATH)

        # create OARs mask based on origin argument and get oar name id mapping
        CT_MASK_PATH = os.path.join(args.tmp_folder, "volume_mask.nii.gz")
        DOSE_MASK_PATH = os.path.join(args.tmp_folder, "dose_mask.nii.gz")

        if args.oar_source == "totalsegmentator":
            print("computing segmentations with TotalSegmentator...")
            oars = create_mask_totalsegmentator(ct, CT_NII_PATH, CT_MASK_PATH)
            
            # same mask for CT and RTDOSE if TotalSegmentator is applied
            DOSE_MASK_PATH = CT_MASK_PATH
        elif args.oar_source == "original":
            print("creating OARs masks based on original contours...")
            oars = create_mask_original(ct, CT_MASK_PATH, args.oar_names, int(p.id))
            create_mask_original(ct.rtdose, DOSE_MASK_PATH, args.oar_names, int(p.id))
        else:
            raise ValueError("argument --oar_source must be one of [original, totalsegmentator]")
        
        if args.filterDA:
            index = get_DA_slices_index(CT_NII_PATH)
            if index:
                remove_slice(index, CT_MASK_PATH)
                remove_slice(index, DOSE_MASK_PATH)
        
        if args.radiomics:
            print("computing radiomics...")
            fts = compute_radiomics_generic(CT_NII_PATH, CT_MASK_PATH, oars, args.radiomics)
            for d in fts:
                d.update({"features": "radiomics", "patient": str(p.id)})
            features.extend(fts)

        if args.dosiomics and ct.rtdose:
            print("computing dosiomics...")
            fts = compute_radiomics_generic(DOSE_NII_PATH, DOSE_MASK_PATH, oars, args.dosiomics)
            for d in fts:
                d.update({"features": "dosiomics", "patient": str(p.id)})
            features.extend(fts)

        if args.dvh and ct.rtdose:
            print("computing DVH features...")

            # Create new RT Struct.
            # Requires the DICOM series path for the RT Struct.
            rtstruct = RTStructBuilder.create_new(dicom_series_path=ct.path)
                    
            # read mask
            mask = sitk.GetArrayFromImage(sitk.ReadImage(CT_MASK_PATH))
            
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
                    features.append({"oar": oar_name, "name": k, "value": v, "features": "dvh", "patient": str(p.id)})

        if args.deepNN:
            print("computing deep nn features...")            

            # read mask
            mask = sitk.GetArrayFromImage(sitk.ReadImage(CT_MASK_PATH))

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
                elif args.deepNN == "fmcib":
                    model = fmcib_model()

                    # see https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/blob/1f1c0c8725c110c9c70cb466467a55e3160760c9/fmcib/preprocessing/__init__.py#L18
                    preprocess = monai_transforms.Compose(
                        [
                            monai_transforms.LoadImage(image_only=True, reader="ITKReader"),
                            monai_transforms.EnsureChannelFirst(),
                            monai_transforms.NormalizeIntensity(subtrahend=-1024, divisor=3072),
                            monai_transforms.Spacing(pixdim=1, padding_mode="zeros", mode="linear", align_corners=True, diagonal=True),
                            monai_transforms.Orientation(axcodes="LPS"),
                            monai_transforms.CropForeground(),   # better than CenterCrop to avoid removing roi
                            monai_transforms.Transpose(indices=(0, 3, 2, 1)),
                            monai_transforms.SpatialPad(spatial_size=(100, 100, 100)),
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
                    features.append({"oar": oar_name, "name": i, "value": f, "features": f"deepnn({args.deepNN})", "patient": str(p.id)})

    # save to csv
    pandas.DataFrame(features).to_csv(args.output)
    print("Done.")
