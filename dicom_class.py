from abc import ABC
import os
from datetime import datetime
import numpy as np
import SimpleITK as sitk
from difflib import SequenceMatcher
import pydicom
import nibabel
import dicom2nifti
from totalsegmentator.python_api import totalsegmentator

from dicom_utils import fill_vol_ctrs


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


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
        elif self.path.endswith(".dcm") or self.path.endswith(".dicom"):
            return self.path
        elif self.path.endswith(".nii.gz"):   # handle HECKTOR 2025 data
            return self.path
        else:
            raise FileNotFoundError(f"DICOM file not found with path {self.path}")
        
    def get_sitk_image(self, dicom_serie=True):
        if dicom_serie:
            try:
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(reader.GetGDCMSeriesFileNames(self.path))
                img = reader.Execute()
            except RuntimeError:
                img  = sitk.ReadImage(self.get_dcm_path())
        else:
            img  = sitk.ReadImage(self.get_dcm_path())
        
        img = sitk.DICOMOrient(img, 'LAS')
        return img

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
        except Exception:
            try:
                return dcm[0x3006,0x0010].value[0][0x0020,0x0052].value
            except Exception:
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
            except Exception:
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
            except Exception:
                print(f"Tag StudyID not available for {path}")
                return None
            except Exception:
                print(f"Tag StudyID not available for {path}")
                return None
            
    def get_acquisition_date(self):
        """
        Return the date of acquisition of the data
        This should be used to know when the image was acquired
        """

        path = self.get_dcm_path()

        if path.endswith(".nii.gz"):   # handle HECKTOR 2025 data
            return None

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

        if not(path_.endswith(".nii.gz")):
            path_ += ".nii.gz"

        # convert DICOM to Nifti
        dicom2nifti.dicom_series_to_nifti(self.path, path_, reorient_nifti=True)


class Imaging(DICOM):
    def __init__(self, path):
        super().__init__(path)

    def get_sitk_image(self):
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(self.path))
        img = reader.Execute()
        return img


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
        if not(self.rtstruct is None) and not(log is None):
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

        # apply TotalSegmentator
        output = totalsegmentator(tmp_nii_input, task=task, quiet=True, verbose=False)

        # return result or save into tmp_nii_output
        if tmp_nii_output is None:
            return output
        else:
            nibabel.save(output, tmp_nii_output)

    def get_GTV_bbox(self):
        if self.get_dcm_path().endswith(".dcm") or self.get_dcm_path().endswith(".dicom"):
            if self.rtstruct is None:
                bbox = None
            else:
                try:
                    # list all contours available in RTSTRUCT
                    seg = self.rtstruct.get_all_OARs()
                    # calculate similarity between contours name and 'gtv'
                    pairs =  [(i, similar("gtv", i.lower().replace(" ", ""))) for i in seg if "gtv" in i.lower()]
                    # select contour with name that most resemble 'gtv'
                    seg, _ = sorted(pairs, reverse=True, key=lambda i: i[1])[0]
                    # get contours coordinates in pixel space
                    ctrs = self.rtstruct.get_contours(seg)
                    # calculate bbox upper-left and lower-right coordinates
                    bbox = ctrs.min(axis=0), ctrs.max(axis=0)
                except IndexError:
                    bbox = None
        elif self.get_dcm_path().endswith(".nii.gz"):   # HECKTOR
            if self.rtstruct is None:
                bbox = None
            else:
                # get GTV bbox coordinates based on segmentation mask
                mask = np.array(nibabel.load(self.rtstruct.path).dataobj)
                gtv = np.argwhere(mask == 1)
                bbox = gtv.min(axis=0), gtv.max(axis=0)
        else:
            bbox = None
        
        return bbox


class RTDOSE(DICOM):
    def __init__(self, path):
        super().__init__(path)

    def get_sitk_image(self):
        img = super().get_sitk_image(dicom_serie=False)
        dcm = pydicom.dcmread(self.get_dcm_path())
        DoseGridScaling = float(dcm.DoseGridScaling)
        dose_array = sitk.GetArrayFromImage(img) * DoseGridScaling
        dose_image = sitk.GetImageFromArray(dose_array)
        dose_image.CopyInformation(img)
        return dose_image

    def convert2nifti(self, path_, align_to_ct=True):
        """
        Convert DICOM data into a Nifti file while adjusting dose values

        args:
            path_ (str) path to nii file to save data
            align_to_ct (bool) weiher to align the volume to CT
        """
        reader = sitk.ImageFileReader()
        reader.SetFileName(self.get_dcm_path())
        reader.ReadImageInformation()

        DoseGridScaling = float(reader.GetMetaData("3004|000e"))

        dose = reader.Execute()

        # IMPORTANT: to be compatible with the other data
        ##################################################################
        dose = sitk.DICOMOrient(dose, 'LAS')
        ##################################################################

        scaled_dose = sitk.GetArrayFromImage(dose) * DoseGridScaling
        scaled_dose = sitk.GetImageFromArray(scaled_dose)
        scaled_dose.CopyInformation(dose)

        if align_to_ct:
            # Resample dose to match CT volume
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(self.parent.get_sitk_image())
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)  # In case resampling introduces new pixels
            scaled_dose = resampler.Execute(scaled_dose)

        sitk.WriteImage(scaled_dose, path_)

    def get_GTV_dose(self):
        def str_similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()
        
        if not(self.parent and self.parent.rtstruct):
            # print("WARNING: cannot return GTV dose as parent is not defined for reference to RTSTRUCT")
            return None

        try:
            # list all contours available in RTSTRUCT
            # calculate similarity between contours name and 'gtv'
            # select contour with name that most resemble 'gtv'
            # get contours coordinates in pixel space
            seg = self.parent.rtstruct.get_all_OARs()
            pairs =  [(i, str_similarity("gtv", i.lower().replace(" ", ""))) for i in seg if "gtv" in i.lower()]
            seg, _ = sorted(pairs, reverse=True, key=lambda i: i[1])[0]
            ctrs = self.parent.rtstruct.get_contours(seg, convert_to_voxel=False)
        except IndexError:
            # print("WARNING: cannot return GTV dose as GTV contour not found in RTSTRUCT")
            return None
        
        if ctrs is None:
            return None

        # transform contours coordinates to RTDOSE voxel space
        rtdose = self.get_sitk_image()
        ctrs = np.array(list(map(rtdose.TransformPhysicalPointToIndex, ctrs.tolist())), dtype=np.int64)

        # create mask of GTV volume
        array = sitk.GetArrayFromImage(rtdose)
        mask = fill_vol_ctrs(array.shape, ctrs)

        # calculate median dose in GTV volume
        gtv_dose = np.median(array[mask])
        return gtv_dose


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
            #  (x, y, z)
            roi_index = {roi.ROIName: roi.ROINumber for roi in dcm.StructureSetROISequence}[name]
            ctrsequence = {item.ReferencedROINumber: item for item in dcm.ROIContourSequence}[roi_index].ContourSequence
            contours = [np.array(ctr.ContourData).reshape(-1, 3) for ctr in ctrsequence] # (x,y,z)
            original_ctr = np.concatenate(contours, axis=0)
        except Exception:
            original_ctr = None

        if original_ctr is None:
            return None
        else:
            original_ctr = np.asarray(original_ctr, dtype="int64")
            if convert_to_voxel:
                if self.parent is None:
                    print("WARNING: cannot return contours as parent is not defined for reference to voxel space affine transformation")
                    return None
                
                ct_image = self.parent.get_sitk_image()                
                original_ctr = np.array(list(map(ct_image.TransformPhysicalPointToIndex, original_ctr.tolist())), dtype=np.int64)
            
            return original_ctr
        
    def get_structure_mask(self, name):
        """
        Return the volume mask of an OAR

        Args:
            name (str) name of structure as defined in the DICOM
        """
        ctrs = self.get_contours(name)
        return fill_vol_ctrs(self.parent.get_shape(), ctrs)

        
class Patient:
    def __init__(self, patient_id, center_id=None, ct=[], cbct=[], clinical={}):
        """
        Create a RT patient with imaging and clinical data

        Args:
            patient_id (int,str) patient ID
            center_id (int,str) center ID
            ct (List) list of CT imaging (including dose and struct)
            cbct (List) list of CBCT imaging
            clinical (dict) dictionnary containing clinical data of patient (age, sexe, ...)
        """
        self.id = patient_id
        self.center_id = center_id
        self.ct = ct
        self.cbct = cbct
        self.clinical = clinical

    def sort_imaging(self):
        """
        Sort images (CT,CBCT) based on acquisition date. 
        Images without this field are placed in the end of the sorted list.
        """

        def safe_sorting_date(i):
            i = list(filter(lambda j: not(j is None), i)) # remove None elements
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
