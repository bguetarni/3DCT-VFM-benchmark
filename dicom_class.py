from abc import ABC
import subprocess
import os, gc, glob, pathlib, shutil
from datetime import datetime
import numpy as np
import dicom2nifti
import SimpleITK as sitk
import pydicom
import nibabel
from nibabel import orientations
from totalsegmentator.python_api import totalsegmentator

from dicom_utils import create_affine, fill_vol_ctrs, convert_ctr_to_voxel_space

"""
CT and RTDOSE/RTSTRUCT files can be matched based on the DICOM tag (0020,0052) Frame of Reference UID
"""

class DICOM(ABC):
    def __init__(self, path, study_base_path=None, parent=None):
        """
        Args:
            path (str) path towards folder containing DICOM files or a DICOM file itself
        """
        self.path = path
        self.study_base_path = study_base_path

        # this propriety should not be used for CT/CBCT !
        self.parent = parent

    def update_study_base_path(self, path):
        """
        Update base path of location of study

        Args:
            path (str) new study path
        """
        
        if not(self.study_base_path is None):
            # reconstruct path by adding study base path to current path minus previous base path
            self.path = os.path.join(path, os.path.relpath(self.path, start=self.study_base_path))
        self.study_base_path = path
    
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
            return datetime.strptime(dcm[0x0008,0x0012].value, '%Y%m%d')
        except (KeyError,AttributeError):
            print(f"Tag Instance Creation Date (0008,0012) not available for {path}")
            return None


class Imaging(DICOM):
    def __init__(self, path):
        super().__init__(path)
        self.nii = None

    def convert2nifti(self, path_):
        """
        Convert DICOM data into a Nifti file

        args:
            path_ (str) path to nii file to save data
        """
        if os.path.isfile(path_):
            os.remove(path_)
        
        # convert DICOM to Nifti
        subprocess.call(["dcm2niix", "-z", "y", "-o", pathlib.Path(path_).parent, "-f", pathlib.Path(pathlib.Path(path_).stem).stem, self.path])
        
        # save path
        self.nii = path_

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

        if not(self.rtstruct is None):
            self.rtstruct.set_parent(self)

        if not(self.rtdose is None):
            self.rtdose.set_parent(self)

    def update_study_base_path(self, path):
        """
        Args:
            path (str) new study path
        """
        super().update_study_base_path(path)

        if not self.rtdose is None:
            self.rtdose.update_study_base_path(path)

        if not self.rtstruct is None:
            self.rtstruct.update_study_base_path(path)
        
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

    def get_shape(self):
        # Get list of DICOM files
        dicom_files = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        
        if not dicom_files:
            return None

        # Read the first DICOM file to get Rows and Columns
        ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)

        return (ds.Rows, ds.Columns, len(dicom_files))

    def apply_totalsegmentator(self, task, tmp_nii_input, tmp_nii_output=None):
        """
        Apply TotalSegmentator model

        args:
            task (str) task to segment (e.g., head_glands_cavities, headneck_muscles, craniofacial_structures)
            tmp_nii_input (str) path to which the converted Nifti data will be saved
            tmp_nii_output (str) path to save the output of segmentation (Nifti mask), if given then this path will be returned, otherwise the output of TotalSegmentator
        """
        # convert DICOM to Nifti
        if self.nii is None:
            self.convert2nifti(tmp_nii_input)
            self.nii = tmp_nii_input
        else:
            shutil.copy(self.nii, tmp_nii_input)

        # apply TotalSegmentator
        output = totalsegmentator(tmp_nii_input, task=task)
        if not(tmp_nii_output is None):
            nibabel.save(output, tmp_nii_output)
            return tmp_nii_output
        else:
            return output.get_fdata()
    
    def gather_contours(self, parotid=True, submandibular=True, mandibule=True, use_totalsegmentator=True, tol=0.1):
        """"
        Gather contours from RTSTRUCT or using TotalSegmentator
        Return dict containing OARs contours with name

        Args:
            parotid (bool) if gathering parotid glands contours
            submandibular (bool) if gathering submandibular glands contours
            use_totalsegmentator (bool) if True use TotalSegmentator to create missing contours
            tol (float) threshold between RTSTRUCT contour and TotalSegmentator to considerate RTSTRUCT 
        """
        pass #TODO


class RTDOSE(DICOM):
    def __init__(self, path):
        super().__init__(path)
    
    def get_voxel_array(self):
        """
        Return dose voxel data
        """

        # raw DICOM voxel values must be scaled by DoseGridScaling to obtain dose values
        DoseGridScaling = pydicom.dcmread(self.get_dcm_path()).DoseGridScaling

        img = sitk.ReadImage(self.get_dcm_path())
        return sitk.GetArrayFromImage(img) * DoseGridScaling

    def get_structure_mask(self, name):
        """
        Return mask of contour on dose volume with shape (H,W,C)

        Args:
            name (str) name of contour as found in RTSTRUCT
        """
        assert not(self.parent is None), f"parent must be initialized first to compute mask of OAR {name}"
        assert not(self.parent.rtstruct is None), f"parent.rtstruct must be initialized first to compute mask of OAR {name}"

        # load contours of structure
        contours = self.parent.rtstruct.get_contours(name, convert_to_voxel=False)
        contours = np.array(contours).reshape(-1, 3)

        # convert contours to scpace
        rtdose_image = sitk.ReadImage(self.get_dcm_path())
        contours = np.array(list(map(rtdose_image.TransformPhysicalPointToIndex, contours)), dtype=np.int64)

        # create mask
        mask = fill_vol_ctrs(sitk.GetArrayFromImage(rtdose_image).shape, contours)

        # if z axis is inverted flip mask
        if rtdose_image.GetDirection()[-1] < 0:
            mask = np.flip(mask, axis=0)
        
        return mask

class RTSTRUCT(DICOM):
    def __init__(self, path):
        super().__init__(path)

    def get_all_OARs(self):
        """
        Return the name of all OARs in the DICOM file
        """
        dcm = pydicom.dcmread(self.get_dcm_path())
        return [roi.ROIName for roi in dcm.StructureSetROISequence]

    def get_contours(self, name, convert_to_voxel=True):
        """
        Return the contours of structure, if ContourSequence not available return None

        Args:
            name (str) name of structure as defined in the DICOM
            convert_to_voxel (bool) if TRUE convert coordinates into voxel space
        """
        
        # read DICOM file
        dcm = pydicom.dcmread(self.get_dcm_path())

        # gather all DICOM structures name and id
        roi_names = {roi.ROINumber: roi.ROIName for roi in dcm.StructureSetROISequence}

        # gather structure coordinates
        original_ctr = []
        for roi_contour in dcm.ROIContourSequence:
            if roi_names[roi_contour.ReferencedROINumber] == name:

                # ContourSequence is a Type 3 property, so not always existing
                if hasattr(roi_contour, "ContourSequence"):
                    for contour in roi_contour.ContourSequence:
                        coords = contour.ContourData
                        # (x0, y0, z0, x1, y1, z1, ...)
                        points = list(zip(coords[0::3], coords[1::3], coords[2::3]))
                        original_ctr.extend(points)
                    break

        if original_ctr:
            original_ctr = np.asarray(original_ctr, dtype="int64")
            if convert_to_voxel:
                if self.parent is None:
                    print("WARNING: cannot return contours as parent is not defined for reference to voxel space affine transformation")
                    return None
                
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(reader.GetGDCMSeriesFileNames(self.parent.path))
                ct_image = reader.Execute()
                original_ctr = np.array(list(map(ct_image.TransformPhysicalPointToIndex, original_ctr)), dtype=np.int64)
            
            return original_ctr
        else:
            return None
        
    def get_structure_mask(self, name):
        """
        Return the volume mask of an OAR

        Args:
            name (str) name of structure as defined in the DICOM
        """
        ctrs = self.get_contours(name)
        return fill_vol_ctrs(self.parent.get_shape(), ctrs)


class Patient:
    def __init__(self, id_, ct=[], cbct=[], clinical={}, clinical_measurements=[]):
        """
        Create a RT patient with imaging and clinical data

        Args:
            id_ (int) patient ID
            ct (List) list of CT imaging (including dose and struct)
            cbct (List) list of CBCT imaging
            clinical (dict) dictionnary containing clinical data of patient (age, sexe, ...)
            clinical_measurements (List) list containing clinical measurements (SSF, DOSIMETRY, MDASI, ...)
        """
        self.id = id_
        self.ct = ct
        self.cbct = cbct
        self.clinical = clinical
        self.clinical_measurements = clinical_measurements

    def sort_imaging(self):
        self.ct = sorted(self.ct, key=lambda x: x.get_acquisition_date())
        self.cbct = sorted(self.cbct, key=lambda x: x.get_acquisition_date())

    def update_study_base_path(self, study_base_path):
        for ct in self.ct:
            ct.update_study_base_path(study_base_path)

        for cbct in self.cbct:
            cbct.update_study_base_path(study_base_path)
