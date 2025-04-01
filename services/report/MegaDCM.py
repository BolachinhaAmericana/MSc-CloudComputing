import pydicom
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset

import random
import numpy as np
import cv2

import matplotlib.pyplot as plt


class UseDicom:
    def __init__(self, dcm_path):
        self.dcm = dcmread(dcm_path)

    def get_metadata(self):
        dcm = self.dcm
        dcm_meta = {}

        for element in dcm:
            if element.VR != 'SQ':  # Skip sequence
                dcm_meta[element.keyword] = str(element.value)

        metadata = {
            "patient_id": dcm_meta['PatientID'],
            "patient_name": dcm_meta['PatientName'],
            "infer_results": dcm_meta['ImageComments'],
            "infer_confidence": dcm_meta['StudyDescription'],
            "obs": dcm_meta['SeriesDescription'],
        }
        return metadata
    
    def get_image_pixel_array(self):
        dcm_file = self.dcm

        if not hasattr(dcm_file, 'file_meta'):
            dcm_file.file_meta = FileMetaDataset()
            dcm_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        img = dcm_file.pixel_array
        return img
    
    def display_image(self, image_pixel_array):
        plt.figure(figsize=(10, 8))
        plt.imshow(image_pixel_array, cmap='gray', vmin=0, vmax=image_pixel_array.max())
        plt.axis('off')
        plt.show()

class GenerateDicom:
    def __init__(self, output_path):
        self.output_path = output_path

        self.file_meta = FileMetaDataset()
        self.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1'  # CR Image Storage
        self.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        self.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        self.ds = FileDataset(output_path, {}, file_meta=self.file_meta, preamble=b"\0"*128)

    def set_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError("Invalid image path or format.")

        if image.dtype != np.uint16:
            image = cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
            ds = self.ds

            # Image Data Attributes
            ds.Rows = image.shape[0]
            ds.Columns = image.shape[1]
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0  # unsigned
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelData = image.tobytes()  # Now properly sized 16-bit data
            self.ds = ds
    
    def set_metadata(self, metadata):
        ds = self.ds

        ds.PatientID = str(metadata.get("patient_id", ""))
        ds.PatientName = metadata.get("patient_name", "")
        ds.ImageComments = str(metadata.get("inference_result", ""))
        ds.StudyDescription = str(metadata.get("inference_confidence", ""))
        ds.SeriesDescription = str(metadata.get("obs", ""))

        ds.Modality = "CR"
        ds.SOPClassUID = self.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = self.file_meta.MediaStorageSOPInstanceUID
        self.ds = ds

    def save(self):
        self.ds.save_as(self.output_path)

    def get_dcm(self):
        return self.ds

if __name__ == "__main__":

    ## Generate and Display Example
    template_metadata = {
            "patient_id": random.randint(1000, 9999),
            "patient_name": random.choice(["Joao", "Maria", "Pedro", "Rita"]),
            "inference_result": random.choice([0,1]),
            "inference_confidence": 0.5,
            "obs": ""
            }

    image_path='../../data/val/NORMAL/NORMAL2-IM-1427-0001.jpeg'
    output_path='./temp.dcm'

    new_dcm = GenerateDicom(output_path=output_path)
    new_dcm.set_image(image_path=image_path)
    new_dcm.set_metadata(metadata=template_metadata)
    new_dcm.save()


    dcm = UseDicom(output_path)
    img = dcm.get_image_pixel_array()
    print(dcm.get_metadata())
    dcm.display_image(img)

