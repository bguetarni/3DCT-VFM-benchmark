import numpy as np
import cv2 as cv
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon2mask

def fill_vol_ctrs(shape, ctrs, use_opencv=True, fill_holes=False):
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
        
        # select points on place
        slice_points = ctrs[ctrs[:,-1] == z]

        # check at least 3 points for polygon
        if slice_points.shape[0] < 3:
            continue

        # fill polygon
        if use_opencv:
            slide_mask = np.zeros(mask.shape[1:], dtype=np.uint8)
            cv.fillPoly(slide_mask, [slice_points[:,:2]], color=1)
            mask[z] = slide_mask
        else:
            # invert slice_points coordinates from (x,y) to (row,column) for polygon2mask
            slide_mask = polygon2mask(mask.shape[1:], slice_points[:,:2][:,::-1])
            mask[z] = slide_mask

    # fill holes in the mask
    # deactivate to run faster
    # if shapes are not filled properly in mask, radiomics might crash !!!
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
