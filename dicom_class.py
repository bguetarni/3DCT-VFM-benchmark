from abc import ABC
import subprocess
import os, pathlib
from datetime import datetime
import numpy as np
import SimpleITK as sitk
import pydicom
import nibabel
from totalsegmentator.python_api import totalsegmentator

from dicom_utils import fill_vol_ctrs


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
            return datetime.strptime(dcm[0x0008,0x0023].value, '%Y%m%d')
        except Exception:
            try:
                return datetime.strptime(dcm[0x0008,0x0022].value, '%Y%m%d')
            except Exception:
                try:
                    return datetime.strptime(dcm[0x0008,0x0012].value, '%Y%m%d')
                except Exception:
                    print(f"None of the following tags are available for {path}: (Content Date Attribute (0008,0023), \
                        Acquisition Date Attribute (0008,0022), Tag Instance Creation Date (0008,0022))")
            return None
        
    def convert2nifti(self, path_):
        """
        Convert DICOM data into a Nifti file

        args:
            path_ (str) path to nii file to save data
        """

        # if file already exists, remove it so dcm2niix can overwrite
        if os.path.isfile(path_):
            os.remove(path_)

        # remove extension as dcm2niix adds it by default
        if path_.endswith(".nii.gz"):
            path_ = path_.removesuffix(".nii.gz")

        # convert DICOM to Nifti
        subprocess.call(["dcm2niix", "-z", "y", "-v", "0", "-o", pathlib.Path(path_).parent, "-f", pathlib.Path(path_).name, self.path])


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

        super().convert2nifti(path_)
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
        """
        Return shape of scan image as (Z,H,W) where Z is the depth, H the height and W the width
        """

        # Get list of DICOM files
        dicom_files = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        
        if not dicom_files:
            return None

        # Read the first DICOM file to get Rows and Columns
        ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)

        return (len(dicom_files), ds.Rows, ds.Columns)

    def apply_totalsegmentator(self, task, tmp_nii_input, tmp_nii_output=None, overwrite_nifti=False):
        """
        Apply TotalSegmentator model

        args:
            task (str) task to segment (e.g., head_glands_cavities, headneck_muscles, craniofacial_structures)
            tmp_nii_input (str) path where Nifti data is (use convert2nifti to convert DICOM to Nifti)
            tmp_nii_output (str) path to save the output of segmentation (Nifti mask), if given then this path will be returned, otherwise the output of TotalSegmentator
            overwrite_nifti (bool) can be used to overwrite Nifti data

        if tmp_nii_output is None, return segmentation
        """

        if overwrite_nifti:
            # if file exists remove it
            if os.path.exists(tmp_nii_input):
                os.remove(tmp_nii_input)

            self.convert2nifti(tmp_nii_input)
            self.nii = tmp_nii_input

        # apply TotalSegmentator
        output = totalsegmentator(tmp_nii_input, task=task, quiet=True, verbose=False)

        # return result or save into tmp_nii_output
        if tmp_nii_output is None:
            return output
        else:
            nibabel.save(output, tmp_nii_output)


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
    
    def convert2nifti(self, path_):
        """
        Convert DICOM data into a Nifti file and adjust dose values

        args:
            path_ (str) path to nii file to save data
        """

        super().convert2nifti(path_)

        # apply scaling for valid dose values
        rtdose_image = sitk.ReadImage(path_)
        dose_array = sitk.GetArrayFromImage(rtdose_image).astype("float")
        dose_array *= float(pydicom.dcmread(self.get_dcm_path()).DoseGridScaling)
        rtdose_image_scaled = sitk.GetImageFromArray(dose_array)
        rtdose_image_scaled.CopyInformation(rtdose_image)
        sitk.WriteImage(rtdose_image_scaled, path_)


class RTSTRUCT(DICOM):
    def __init__(self, path):
        super().__init__(path)

    def get_all_OARs(self):
        """
        Return the name of all OARs in the DICOM file
        """
        dcm = pydicom.dcmread(self.get_dcm_path())
        return [roi.ROIName for roi in dcm.StructureSetROISequence]
    
    def get_OAR_id(self, name):
        """
        Return the id of OAR name as found in DICOM

        args
            name (str) name of OAR as found in DICOM RTSTRUCT
        """
        dcm = pydicom.dcmread(self.get_dcm_path())
        return {roi.ROIName: roi.ROINumber for roi in dcm.StructureSetROISequence}[name]

    def get_contours(self, name, convert_to_voxel=True):
        """
        Return the contours of structure, if ContourSequence not available return None

        Args:
            name (str) name of structure as defined in the DICOM
            convert_to_voxel (bool) if TRUE convert coordinates into voxel space
        """
        
        # read DICOM file
        dcm = pydicom.dcmread(self.get_dcm_path())

        # gather structure coordinates
        try:
            roi_index = {roi.ROIName: roi.ROINumber for roi in dcm.StructureSetROISequence}[name]
            ctrsequence = {item.ReferencedROINumber: item for item in dcm.ROIContourSequence}[roi_index].ContourSequence
            contours = [np.array(ctr.ContourData).reshape(-1, 3) for ctr in ctrsequence] # (x,y,z)
            original_ctr = np.concatenate(contours, axis=0)
        except Exception:
            original_ctr = None

        if not(original_ctr is None):
            original_ctr = np.asarray(original_ctr, dtype="int64")
            if convert_to_voxel:
                if self.parent is None:
                    print("WARNING: cannot return contours as parent is not defined for reference to voxel space affine transformation")
                    return None
                
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(reader.GetGDCMSeriesFileNames(self.parent.path))
                ct_image = reader.Execute()
                original_ctr = np.array(list(map(ct_image.TransformPhysicalPointToIndex, original_ctr.tolist())), dtype=np.int64)
            
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
    
    def create_nii_mask_oars(self, path_=None, oars=None):
        """
        Create a mask with labels as found in DICOM file similar to TotalSegmentator labeling.
        If path_ is provided, save it into as a Nifti image

        args
            path_ (str) path to nifti file, if None return mask
            oars (List[str]) list of oars to compute mask, if None compute for all OARs in RTSTRUCT
        """

        # initialize empty mask
        mask = np.zeros(shape=self.parent.get_shape(), dtype=np.uint8)
        
        # read DICOM file
        dcm = pydicom.dcmread(self.get_dcm_path())
        
        try:
            for roi in dcm.StructureSetROISequence:
                if isinstance(oars, list) and not(roi.ROIName in oars):
                    continue

                roi_mask = self.get_structure_mask(roi.ROIName)
                mask[roi_mask] = roi.ROINumber
        except Exception:
            return None
        
        if path_:
            sitk.WriteImage(sitk.GetImageFromArray(mask), path_)
        else:
            return mask
        
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
        """
        Sort images (CT,CBCT) based on acquisition date. 
        Images without this field are placed in the end of the sorted list.
        """

        def safe_sorting_date(i):
            valid_date = filter(lambda j: not(j.get_acquisition_date() is None), i)
            not_valid_date = filter(lambda j: j.get_acquisition_date() is None, i)
            return [*sorted(valid_date, key=lambda j: j.get_acquisition_date()), *not_valid_date]

        self.ct = safe_sorting_date(self.ct)
        self.cbct = safe_sorting_date(self.cbct)
        
    def update_study_base_path(self, study_base_path):
        for ct in self.ct:
            ct.update_study_base_path(study_base_path)

        for cbct in self.cbct:
            cbct.update_study_base_path(study_base_path)
