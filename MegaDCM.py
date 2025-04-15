from flask import Flask, request, jsonify, send_from_directory
import grpc
import io
from PIL import Image
import numpy as np
import pydicom
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
import cv2
import matplotlib.pyplot as plt
import tempfile
import os
import preprocessing_pb2
import preprocessing_pb2_grpc
import inference_pb2
import inference_pb2_grpc
import reporting_pb2
import reporting_pb2_grpc

# Initialize Flask app
app = Flask(__name__)


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
    '''
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
    '''
    
    
    
# Now let's handle the workflow of the app

# First by making functions to call the grpc services
def call_preprocessing_service(image_bytes):
    channel = grpc.insecure_channel('preprocessing:5050')
    stub = preprocessing_pb2_grpc.PreprocessingServiceStub(channel)
    request = preprocessing_pb2.ImageRequest(image_data=image_bytes)
    response = stub.ProcessImage(request)
    return response.tensor_data


def call_inference_service(tensor_data):
    channel = grpc.insecure_channel('inference:50051')
    stub = inference_pb2_grpc.InferenceServiceStub(channel)
    request = inference_pb2.InferenceRequest(tensor_data=tensor_data)
    response = stub.Predict(request)
    return response.predicted_class, response.confidence

def call_reporting_service(predicted_class,confidence, metadata):
    channel = grpc.insecure_channel('reporting:5052')
    stub = reporting_pb2_grpc.ReportingServiceStub(channel)
    request = reporting_pb2.ReportRequest(
        predicted_class=predicted_class,
        confidence=confidence,
        metadata=str(metadata)
    )
    
    response = stub.GenerateReport(request)
    return response.report_url


# Flask Routes
@app.route('/')

def serve_index():
    return send_from_directory('static','index.html') # podemos alterar os nomes desta pasta e ficheiro depois

@app.route('/upload-dicom', methods=['POST'])

#Let's first receive the DICOM  image from the client side
def upload_dicom():
    if 'dicom' not in request.files:
        return jsonify({'error': 'No DICOM  file provided'}), 400
    dicom_file = request.files['dicom']
    dicom_data = dicom_file.read()
    
    # Like we did with the model we will store the file temporarily to process with UseDIcom class
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_dcm:
        temp_dcm.write(dicom_data)
        temp_dcm_path = temp_dcm.name
    
    try:
        dicom_processor = UseDicom(temp_dcm_path)
        image_array = dicom_processor.get_image_pixel_array()
        metadata = dicom_processor.get_metadata()
        
        image = Image.fromarray(image_array).convert('L')
        image_buffer = io.BytesIO()
        image.save(image_buffer,format='PNG')
        image_bytes = image_buffer.getvalue()
        
    except Exception as e:
        os.remove(temp_dcm_path)
        return jsonify({"error": f"Failed to process DICOM: {str(e)}"}), 500
    finally:
        os.remove(temp_dcm_path) # We clean this to save disk space
    
    # Then Let's process the image by passing trough that microservice
    try:
        tensor_data = call_preprocessing_service(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 500
    
    # After we will pass trough the inference
    try:
        predicted_class, confidence = call_inference_service(tensor_data)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500
    
    # Finally we generate the report
    
    # Update metadata with inference results
    metadata["inference_result"] = predicted_class
    metadata["inference_confidence"] = str(confidence)
    
    
    try:
        report_url = call_reporting_service(predicted_class, confidence, metadata)
    except Exception as e:
        return jsonify({"error": f"Reporting failed: {str(e)}"}), 500
    
    # Return report URL to client
    return jsonify({"report_url": report_url}), 200
    
        
        
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
