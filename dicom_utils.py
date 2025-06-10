import os, pathlib
import numpy as np
import cv2
from dicom2nifti.exceptions import ConversionError

def get_directory_level(path1, path2):
    """
    Return the number of levels that separate each path with the common parent path
    """
    commonpath = os.path.commonpath([path1, path2])
    relpath1 = os.path.relpath(path1, start=commonpath)
    relpath2 = os.path.relpath(path2, start=commonpath)
    return len(pathlib.Path(relpath1).parents), len(pathlib.Path(relpath2).parents)

# modified version of dicom2nifti.common.create_affine
def create_affine(sorted_dicoms):
    """
    Function to generate the affine matrix for a dicom series
    This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)

    :param sorted_dicoms: list with sorted dicom files
    """

    # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
    image_orient1 = np.array(sorted_dicoms[0].ImageOrientationPatient)[0:3]
    image_orient2 = np.array(sorted_dicoms[0].ImageOrientationPatient)[3:6]

    delta_r = float(sorted_dicoms[0].PixelSpacing[0])
    delta_c = float(sorted_dicoms[0].PixelSpacing[1])

    image_pos = np.array(sorted_dicoms[0].ImagePositionPatient)

    last_image_pos = np.array(sorted_dicoms[-1].ImagePositionPatient)

    if len(sorted_dicoms) == 1:
        # Single slice
        slice_thickness = 1
        if "SliceThickness" in sorted_dicoms[0]:
            slice_thickness = sorted_dicoms[0].SliceThickness
        step = - np.cross(image_orient1, image_orient2) * slice_thickness
    else:
        step = (image_pos - last_image_pos) / (1 - len(sorted_dicoms))

    # check if this is actually a volume and not all slices on the same location
    if np.linalg.norm(step) == 0.0:
        raise ConversionError("NOT_A_VOLUME")
    
    ##########################################################################
    ############ MODIFICATION FROM ORIGINAL dicom2nifti VERSION ##############
    # signs of first and second row are inverted (from - to +) except for step
    # delta_r and delta_c are inverted
    affine = np.array(
        [[image_orient1[0] * delta_r, image_orient2[0] * delta_c, -step[0], image_pos[0]],
         [image_orient1[1] * delta_r, image_orient2[1] * delta_c, -step[1], image_pos[1]],
         [image_orient1[2] * delta_r, image_orient2[2] * delta_c, step[2], image_pos[2]],
         [0, 0, 0, 1]]
    )
    ##########################################################################
    
    return affine

def fill_vol_ctrs(shape, ctrs):
    """
    Create mask of shape (*shape) anf fill contours with 1s

    Args:
        shape (tuple) shape of mask
        ctrs (List,tuple) contours arranged as (x,y,z)
    """
    mask = np.zeros(shape, dtype="uint8")
    for z in np.unique(ctrs[:,2]):
        img = np.zeros(shape[:2], dtype="uint8")
        zctrs = ctrs[ctrs[:,2] == z][:,:2]
        cv2.fillPoly(img, [zctrs], 1)
        mask[:,:,z] = img
    
    return mask
