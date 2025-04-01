import pydicom
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset

import random
import numpy as np
import cv2

import matplotlib.pyplot as plt


class DICOM:
    def __init__(self, dcm_path):
        self.dcm_path = dcm_path
        self.dcm = dcmread(dcm_path)

    def generate_dicom(self, image_path, metadata, output_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError("Invalid image path or format.")
            
        # Convert to 16-bit if needed (DICOM standard prefers 16-bit)
        if image.dtype != np.uint16:
            image = cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        
        # Create file meta
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1'  # CR Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0"*128)
        
        # Set required DICOM attributes
        ds.PatientID = str(metadata.get("patient_id", ""))
        ds.PatientName = metadata.get("patient_name", "")
        ds.Modality = "CR"
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        
        # Critical Pixel Data Attributes
        ds.Rows = image.shape[0]
        ds.Columns = image.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # unsigned
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = image.tobytes()  # Now properly sized 16-bit data

        # Save file
        ds.save_as(output_path)
        print(f"DICOM file successfully saved as {output_path}")

    def get_dcmGen_randomMetadata(self):
        name_list = ["Joao", "Maria", "Pedro", "Rita"]

        metadata = {
        "patient_id": random.randint(1000, 9999),
        "patient_name": random.choice(name_list),
        "inference_result": 1,
        "inference_confidence": 0.5,
        "obs": ""
        }
        return metadata

    def get_dicom_image(self):
        dcm_file = self.dcm

        if not hasattr(dcm_file, 'file_meta'):
            dcm_file.file_meta = FileMetaDataset()
            dcm_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        img = dcm_file.pixel_array
        return img
    
    def get_dicom_metadata(self):
        """Extracts relevant metadata from the DICOM dataset."""
        if dcm_file is None:
            dcm_file = self.dcm
        
        metadata = {}
        for element in dcm_file:
            if element.VR != 'SQ':  # Skip sequence
                metadata[element.keyword] = str(element.value)
        return metadata

    def show_dicom_image(self, dcm_file=None):
        """ Show Dicom Image """
        """ Falta um try except para n ter chance de fazer bum"""
        if dcm_file is None:
            dcm_file = self.dcm
    
        img = self.get_dicom_image()

        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray', vmin=0, vmax=img.max())
        plt.title(f"DICOM id: \n{dcm_file.get('PatientID', '')}")
        plt.colorbar(label='Pixel Intensity')
        plt.axis('off')
        plt.show()

        print("\n=== DICOM Stats ===")
        print(f"Dimensions: {dcm_file.Rows}x{dcm_file.Columns}")
        print(f"Pixel data type: {img.dtype}")
        print(f"Expected bytes: {dcm_file.Rows * dcm_file.Columns * (dcm_file.BitsAllocated//8)}")
        print(f"Actual bytes: {len(dcm_file.PixelData)}")

    



    


