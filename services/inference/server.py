import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc
from inference import InferenceHandler
import torch
import logging
import os

class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.handler = InferenceHandler()

    def Predict(self, request, context):
        # Reconstruct the tensor from the request because proto does not support tensors. 
        tensor_data = request.tensor_data
        tensor_shape = (1, 1, 224, 224) # Also we are hardcoding this time because we will not be altering the preprocessing pipeline otherwise variables would be more flexible
        expected_elements = 1 * 1 * 224 * 224 
        if len(tensor_data) != expected_elements:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Expected {expected_elements} elements in tensor_data, got {len(tensor_data)}")
            return inference_pb2.InferenceResponse() #Just for validation
        
        tensor = torch.tensor(tensor_data).reshape(tensor_shape)
        
        
        predicted_class, confidence = self.handler.predict(tensor)
        
        # Return the response to report generator to fetch it
        return inference_pb2.InferenceResponse(
            predicted_class=predicted_class,
            confidence=confidence
        )

def serve():
    # Create a gRPC server with a thread pool
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceService(), server
    )
    server.add_insecure_port('[::]:50051')  # Listen on port 50051
    print("Starting gRPC server on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()