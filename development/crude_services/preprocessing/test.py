import grpc
from preprocessing_pb2 import ProcessRequest
from preprocessing_pb2_grpc import PreprocessorStub

channel = grpc.insecure_channel("localhost:5000")
client = PreprocessorStub(channel)

with open("test_xray.jpeg", "rb") as f:
    image_data = f.read()

response = client.Process(ProcessRequest(
    image_data=image_data,  
    user_id="admin_123"  
))

print("Success! The processed image has been saved to:", response.gcs_path)