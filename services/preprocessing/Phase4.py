from concurrent import futures
import grpc
import logging
from preprocessing_pb2 import ProcessResponse
from preprocessing_pb2_grpc import PreprocessorServicer, add_PreprocessorServicer_to_server
from torchvision.transforms import v2 as transforms
from PIL import Image
import io
import cv2
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account
import torch
import google




class Preprocessor(PreprocessorServicer):  
    def __init__(self, bucket_name="xray-bucket-fcul", project_id="spring-gift-425520-n2"):
        # Preprocess/transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: Image.fromarray(
                cv2.GaussianBlur(np.array(x), (5, 5), 0)
            )),
            transforms.Lambda(lambda x: Image.fromarray(
                cv2.equalizeHist(np.array(x)) if len(np.array(x).shape) == 2
                else cv2.merge([cv2.equalizeHist(ch) for ch in cv2.split(np.array(x))])
            )),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        credentials = service_account.Credentials.from_service_account_file(
            '/app/gcp-credentials.json'  
        )
        self.client = storage.Client(project=project_id, credentials=credentials)
        self.bucket = self.client.bucket(bucket_name)

    def Process(self, request, context):
        try:
            # Step 1: Load and process image
            image_data = request.image_data
            rows = request.rows
            cols = request.cols
            user_id = request.user_id

            img_array = np.frombuffer(image_data, dtype=np.uint16)  
            img_array = img_array.reshape(rows, cols)

            img_array = (img_array / img_array.max() * 255).astype(np.uint8) # PIL could not be directly used
            image = Image.fromarray(img_array)

            processed_tensor = self.transform(image)  # Shape: [1, 224, 224]
            
            # Step 2: Convert processed tensor back to saveable image
            # Remove normalization and scale to 0-255
            img_to_save = processed_tensor.squeeze(0)  # Remove channel dim -> [224, 224]
            img_to_save = (img_to_save * 127.5 + 127.5).byte()  # Denormalize
            processed_pil = Image.fromarray(img_to_save.numpy())
            
            # Step 3: Save processed image to GCS
            gcs_path = f"processed/{user_id}.png"
            blob = self.bucket.blob(gcs_path)
            
            img_byte_arr = io.BytesIO()
            processed_pil.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            blob.upload_from_file(img_byte_arr, content_type='image/png')
            
            # Step 4: Return paths and tensor
            return ProcessResponse(
                gcs_path=f"gs://{self.bucket.name}/{gcs_path}",
                tensor_data=processed_tensor.numpy().tobytes()  
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error: {str(e)}")
            return ProcessResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_PreprocessorServicer_to_server(Preprocessor(), server)
    server.add_insecure_port("[::]:50050")
    print("Starting gRPC server on port 50050...")
    server.start()
    print('Listening... ')
    logging.info("Server started on port 50050")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()