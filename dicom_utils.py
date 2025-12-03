import os
import pathlib
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon
import SimpleITK as sitk

def get_directory_level(path1, path2):
    """
    Return the number of levels that separate each path with the common parent path
    """
    commonpath = os.path.commonpath([path1, path2])
    relpath1 = os.path.relpath(path1, start=commonpath)
    relpath2 = os.path.relpath(path2, start=commonpath)
    return len(pathlib.Path(relpath1).parents), len(pathlib.Path(relpath2).parents)

def fill_vol_ctrs(shape, ctrs, fill_holes=True):
    """
    Create mask of shape (*shape) anf fill contours with 1s

    Args:
        shape (tuple) shape of mask (z,h,w)
        ctrs (List,tuple) contours arranged as (x,y,z)
    """
    mask = np.zeros(shape, dtype=bool)
    for z in np.unique(ctrs[:,-1]):

        # check for boundaries
        if z >= mask.shape[0]:
            continue
        
        # select point on place
        slice_points = ctrs[ctrs[:,-1] == z]

        # check at least 3 points for polygon
        if slice_points.shape[0] < 3:
            continue

        # fill polygon
        rr, cc = polygon(slice_points[:,1], slice_points[:,0], shape=mask.shape[1:])
        mask[z, rr, cc] = 1

    # fill holes in the mask
    # deactivate to run faster
    # if shapes are not filled properly in OAR masks, radiomics might crash !!!
    if fill_holes:
        for z in range(mask.shape[0]):
            mask[z] = binary_fill_holes(mask[z])
    
    return mask


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

def convert_ctr_to_voxel_space(affine, points):
    """
    Convert world space ccordinates into voxel space coordinates based on inverse transformation

    Args:
        affine (numpy.ndarray) affine transformation from voxel to world space
        points (numpy.ndarray) points coordinates to convert (N,3)
    """

    # inverse affine transformation
    inv_affine = np.linalg.inv(affine)

    # homogeneous coordinates
    points = np.hstack((points, np.ones((points.shape[0], 1))))

    # apply affine transformation
    voxel_ctr = inv_affine @ points.transpose()

    # reorder axes
    voxel_ctr = voxel_ctr.transpose().astype("int64")

    # drop homogeneous coordinate
    return voxel_ctr[:,:3]

def resample_dose_to_ct(ct_nii_path, dose_nii_path, out_path):
    # Resample dose to match CT volume
    planning_ct = sitk.ReadImage(ct_nii_path)
    dose_image = sitk.ReadImage(dose_nii_path)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(planning_ct)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)  # In case resampling introduces new pixels
    resampled_dose = resampler.Execute(dose_image)
    sitk.WriteImage(resampled_dose, out_path)
