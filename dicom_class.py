from abc import ABC
import os, gc, glob
from datetime import datetime
import numpy as np
import dicom2nifti
import pydicom
import nibabel
from nibabel import orientations
from totalsegmentator.python_api import totalsegmentator

from dicom_utils import create_affine

"""
CT and RTDOSE/RTSTRUCT files can be matched based on the DICOM tag (0020,0052) Frame of Reference UID
"""

class DICOM(ABC):
    def __init__(self, path):
        """
        Args:
            path (str) path towards folder containing DICOM files or a DICOM file itself
        """
        self.path = path

        # this propriety should not be used for CT/CBCT !
        self.parent = None
    
    def set_parent(self, parent):
        """
        Set new parent which must be a CT
        """
        self.parent = parent
    
    def get_parent(self):
        return self.parent
    
    def get_dcm_path(self):
        if os.path.isdir(self.path):
            # if path is folder, take first file inside
            return os.path.join(self.path, os.listdir(self.path)[0])
        else:
            # else return path itself
            return self.path

    def get_FrameOfReferenceUID(self):
        """
        Return FrameOfReferenceUID tag of DICOM file/folder if available, otherwise return None
        """
        
        path = self.get_dcm_path()
        
        try:
            dcm = pydicom.dcmread(path)
        except PermissionError:
            print(f"WARNING: {path} not accessible with pydicom")
            return None
        
        try:
            return dcm[0x0020,0x0052].value
        except KeyError:
            try:
                return dcm[0x3006,0x0010].value[0][0x0020,0x0052].value
            except AttributeError:
                print(f"Tag FrameOfReferenceUID not available for {path}")
                return None

    def get_StudyInstanceUID(self):
            """
            Return StudyInstanceUID tag of DICOM file/folder if available, otherwise return None
            """
            
            path = self.get_dcm_path()
            
            try:
                dcm = pydicom.dcmread(path)
            except PermissionError:
                print(f"WARNING: {path} not accessible with pydicom")
                return None
            
            try:
                return dcm[0x0020,0x000d].value
            except KeyError:
                print(f"Tag StudyInstanceUID not available for {path}")
                return None
            except AttributeError:
                print(f"Tag StudyInstanceUID not available for {path}")
                return None

    # This SHOULD NOT be used to check if CT and RTSTRUCT/RTDOSE are from same study !!
    def get_StudyID(self):
            """
            Return StudyID tag of DICOM file/folder if available, otherwise return None
            """
            
            path = self.get_dcm_path()
            
            try:
                dcm = pydicom.dcmread(path)
            except PermissionError:
                print(f"WARNING: {path} not accessible with pydicom")
                return None
            
            try:
                return dcm[0x0020,0x0010].value
            except KeyError:
                print(f"Tag StudyID not available for {path}")
                return None
            except AttributeError:
                print(f"Tag StudyID not available for {path}")
                return None
            
    def get_acquisition_date(self):
        """
        Return the date of acuqisition of the data
        This should be sued to knwo when the image was acquired
        """

        path = self.get_dcm_path()

        try:
            dcm = pydicom.dcmread(path)
        except PermissionError:
            print(f"WARNING: {path} not accessible with pydicom")
            return None
        
        try:
            return datetime.strptime(dcm[0x0008,0x0022].value, '%Y%m%d')
        except KeyError:
            print(f"Tag AcquisitionDate (0008,0022) not available for {path}")
            return None
        except AttributeError:
            print(f"Tag AcquisitionDate (0008,0022) not available for {path}")
            return None


class Imaging(DICOM):
    def __init__(self, path):
        super().__init__(path)
        self.nii = None

    def clear_nii(self):
        """
        Call to reduce memory usage
        """
        self.nii = None
        gc.collect()

    def load_nii(self, reorient=True, recalculate_affine=True):
        """
        Load DICOM folder into nifti format with or without reorientation and affine recalculation

        Args:
            reorient (bool) to reorient or not the image into LPS orientation
            recalculate_affine (bool) to recalculate affine transformation with modified version
        """

        # load DICOM folder into nifti object
        self.nii = dicom2nifti.convert_dicom.dicom_series_to_nifti(
            original_dicom_directory=self.path, 
            output_file=None,
            reorient_nifti=False)["NII"]
        
        if reorient or recalculate_affine:
            if reorient:
                # reorient volume and affine transform into (x,y,z) (right left, anterior posterior, inferior superior)
                current_ornt = orientations.io_orientation(self.nii.affine)
                target_ornt = orientations.axcodes2ornt(('P', 'L', 'S'))
                transform = orientations.ornt_transform(current_ornt, target_ornt)
                new_data = orientations.apply_orientation(self.nii.dataobj, transform)
            else:
                new_data = self.nii.dataobj

            if recalculate_affine:
                # create affine transform with corrected sign and deltas
                dcm_files = glob.glob(os.path.join(self.path, "*.dcm"))
                sorted_ = dicom2nifti.common.sort_dicoms(list(map(pydicom.dcmread, dcm_files)))
                new_affine = create_affine(sorted_)
            else:
                new_affine = self.nii.affine

            # build nifti object
            self.nii = nibabel.Nifti1Image(new_data, new_affine)

    def get_voxel_array(self):
        """
        Return image voxel data
        """

        if self.nii is None:
            self.load_nii()
        
        return self.nii.dataobj
        
    def get_affine(self):
        """
        Return affine transformation for machine to voxel space
        """

        if self.nii is None:
            self.load_nii()
        
        return self.nii.affine
    


class CBCT(Imaging):
    def __init__(self, path):
        super().__init__(path)


class CT(Imaging):
    def __init__(self, path, rtdose=None, rtstruct=None):
        """
        Args:
            path (str) path towards folder containing CBCT
            rtdose (str) an RTDOSE object to associate to CT
            rtstruct (str) an RTSTRUCT object to associate to CT
        """
        super().__init__(path)

        # RTDOSE and RTSTRUCT objects associated to this CT
        self.rtdose = rtdose
        self.rtstruct = rtstruct
        
    def add_rtdose(self, rtdose, log=None):
        """
        Add RTDOSE to CT imaging

        Args:
            rtdose (RTDOSE) object to add
            log (logging.Logger) logger for warning messages
        """
        if not self.rtdose is None and not log is None:
            log.warning(f"WARNING: RTDOSE {self.rtdose.path} of CT {self.path} is being replaced by {rtdose.path}")
        self.rtdose = rtdose
        self.rtdose.set_parent(self)

    def add_rtstruct(self, rtstruct, log=None):
        """
        Add RTSTRUCT to CT imaging

        Args:
            rtstruct (RTSTRUCT) object to add
            log (logging.Logger) logger for warning messages
        """
        if not self.rtstruct is None and not log is None:
            log.warning(f"WARNING: RTSTRUCT {self.rtstruct.path} of CT {self.path} is being replaced by {rtstruct.path}")
        self.rtstruct = rtstruct
        self.rtstruct.set_parent(self)

    def apply_totalsegmentator(self):
        self.load_nii(reorient=False, recalculate_affine=False)
        output_img = totalsegmentator(self.nii, task="head_glands_cavities")
        current_ornt = orientations.io_orientation(self.nii.affine)
        target_ornt = orientations.axcodes2ornt(('P', 'L', 'S'))
        transform = orientations.ornt_transform(current_ornt, target_ornt)
        return orientations.apply_orientation(output_img.dataobj, transform)
    
    def gather_contours(self, parotid=True, submandibular=True, use_totalsegmentator=True, tol=0.1):
        """"
        Gather contours from RTSTRUCT or using TotalSegmentator

        Args:
            parotid (bool) if gathering parotid glands contours
            submandibular (bool) if gathering submandibular glands contours
            use_totalsegmentator (bool) if True use TotalSegmentator to create missing contours
            tol (float) threshold between RTSTRUCT contour and TotalSegmentator to considerate RTSTRUCT 
        """
        pass


class RTDOSE(DICOM):
    def __init__(self, path):
        super().__init__(path)


class RTSTRUCT(DICOM):
    def __init__(self, path):
        super().__init__(path)

    def get_all_OARs(self):
        """
        Return the name of all OARs in the DICOM file
        """
        dcm = pydicom.dcmread(self.get_dcm_path())
        return [roi.ROIName for roi in dcm.StructureSetROISequence]

    def convert_ctr_to_voxel_space(self, original_ctr):  
        # get affine transformation matrix      
        affine = self.parent.get_affine()
        
        # inverse affine transformation
        inv_affine = np.linalg.inv(affine)

        # homogeneous coordinates
        original_ctr = np.hstack((original_ctr, np.ones((original_ctr.shape[0], 1))))

        # apply affine transformation
        voxel_ctr = inv_affine @ original_ctr.transpose()

        # reorder axes
        voxel_ctr = voxel_ctr.transpose().astype("int64")

        # drop homogeneous coordinate
        return voxel_ctr[:,:3]

    def get_contours(self, structure_name):
        """
        Return the contours in voxel space of structure, if ContourSequence not available return None

        Args:
            structure_name (str) name of structure as defined in the DICOM
        """

        if self.parent is None:
            print("WARNING: cannot return contours as parent is not defined for reference to voxel space affine transformation")
            return None
        
        # read DICOM file
        dcm = pydicom.dcmread(self.get_dcm_path())

        # gather all DICOM structures name and id
        roi_names = {roi.ROINumber: roi.ROIName for roi in dcm.StructureSetROISequence}

        # gather structure coordinates
        original_ctr = []
        for roi_contour in dcm.ROIContourSequence:
            if roi_names[roi_contour.ReferencedROINumber] == structure_name:

                # ContourSequence is a Type 3 property, so not always existing
                if hasattr(roi_contour, "ContourSequence"):
                    for contour in roi_contour.ContourSequence:
                        coords = contour.ContourData
                        # (x0, y0, z0, x1, y1, z1, ...)
                        points = list(zip(coords[0::3], coords[1::3], coords[2::3]))
                        original_ctr.extend(points)
                    break

        if original_ctr:
            return self.convert_ctr_to_voxel_space(np.asarray(original_ctr, dtype="int64"))
        else:
            return None


class Patient:
    def __init__(self, id_, ct=[], cbct=[], clinical={}):
        """
        Create a RT patient with imaging and clinical data

        Args:
            id_ (int) patient ID
            ct (List) list of CT imaging (including dose and struct)
            cbct (List) list of CBCT imaging
            clinical (dict) dictionnary containing clinical data of patient (age, sexe, ...)
        """
        self.id = id_
        self.ct = ct
        self.cbct = cbct
        self.clinical = clinical

    def sort_imaging(self):
        self.ct = sorted(self.ct, key=lambda x: x.get_acquisition_date())
        self.cbct = sorted(self.cbct, key=lambda x: x.get_acquisition_date())
