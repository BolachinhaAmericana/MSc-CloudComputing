# server.py
import grpc
from concurrent import futures
import time
import numpy as np
import pydicom
from pydicom.dataset import FileMetaDataset
import MegaDCM_pb2
import MegaDCM_pb2_grpc
import cv2

from MegaDCM import UseDicom, GenerateDicom

class DicomServiceServicer(MegaDCM_pb2_grpc.DicomServiceServicer):
    def GetMetadata(self, request, context):
        dicom = UseDicom(request.path)
        metadata = dicom.get_metadata()

        return MegaDCM_pb2.Metadata(
            patient_id=metadata.get("patient_id", ""),
            patient_name=metadata.get("patient_name", ""),
            infer_results=metadata.get("infer_results", ""),
            infer_confidence=metadata.get("infer_confidence", ""),
            obs=metadata.get("obs", "")
        )

    def GetImagePixelArray(self, request, context):
        dicom = UseDicom(request.path)
        img = dicom.get_image_pixel_array()
        flat = img.flatten().tolist()

        return MegaDCM_pb2.ImageArray(
            data=flat,
            rows=img.shape[0],
            cols=img.shape[1]
        )

    def GenerateDicom(self, request, context):
        gen = GenerateDicom(request.output_path)
        gen.set_image(request.image_path)

        meta = {
            "patient_id": request.metadata.patient_id,
            "patient_name": request.metadata.patient_name,
            "inference_result": request.metadata.infer_results,
            "inference_confidence": request.metadata.infer_confidence,
            "obs": request.metadata.obs,
        }

        gen.set_metadata(meta)
        gen.save()

        return MegaDCM_pb2.DicomPath(path=request.output_path)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    MegaDCM_pb2_grpc.add_DicomServiceServicer_to_server(DicomServiceServicer(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("DCM server started on port 50052.")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
