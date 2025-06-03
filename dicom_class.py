import os, pathlib
from datetime import datetime
import pydicom

"""
CT and RTDOSE/RTSTRUCT files can be matched based on the DICOM tag (0020,0052) Frame of Reference UID
"""

class DICOM:
    def __init__(self, path):
        """
        Args:
            path (str) path towards folder containing DICOM files or a DICOM file itself
        """
        self.path = path
    
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

class RTDOSE(DICOM):
    def __init__(self, path):
        """
        Args:
            path (str) path towards folder containing RTDOSE
        """
        super().__init__(path)

class RTSTRUCT(DICOM):
    def __init__(self, path):
        """
        Args:
            path (str) path towards folder containing RTSTRUCT
        """
        super().__init__(path)

class CBCT(DICOM):
    def __init__(self, path):
        """
        Args:
            path (str) path towards folder containing CBCT
        """
        super().__init__(path)

class CT(DICOM):
    def __init__(self, path, rtdose=None, rtstruct=None):
        """
        Args:
            path (str) path towards folder containing CBCT
            rtdose (str) an RTDOSE object to associate to CT
            rtstruct (str) an RTSTRUCT object to associate to CT
        """
        super().__init__(path)
        self.rtdose = rtdose
        self.rtstruct = rtstruct
        
    def add_rtdose(self, rtdose):
        """
        Add RTDOSE to CT imaging

        Args:
            rtdose (RTDOSE) object to add
        """
        if not self.rtdose is None:
            print(f"WARNING: RTDOSE {self.rtdose.path} of CT {self.path} is being replaced by {rtdose.path}")
        self.rtdose = rtdose

    def add_rtstruct(self, rtstruct):
        """
        Add RTSTRUCT to CT imaging

        Args:
            rtstruct (RTSTRUCT) object to add
        """
        if not self.rtstruct is None:
            print(f"WARNING: RTSTRUCT {self.rtstruct.path} of CT {self.path} is being replaced by {rtstruct.path}")
        self.rtstruct = rtstruct


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
