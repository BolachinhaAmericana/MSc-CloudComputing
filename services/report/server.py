import grpc
from concurrent import futures
import time
import os
import tempfile
from PIL import Image
import io
import numpy as np

import report_pb2
import report_pb2_grpc

from reportGenerator import ReportGenerator  # Your existing ReportGenerator class

class ReportService(report_pb2_grpc.ReportServiceServicer):

    def GenerateReport(self, request, context):
        try:
            image_bytes = request.image
            img = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(img)

            # Parse metadata from key-value repeated field
            metadata = {item.key: item.value for item in request.metadata}

            # Determine output file path
            output_filename = request.output_filename or "output.pdf"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)

            # Generate PDF using your ReportGenerator class
            generator = ReportGenerator(img_array, metadata, output_path)
            generator.generate_report()

            return report_pb2.ReportResponse(
                success=True,
                message=f"Report generated at: {output_path}"
            )
        except Exception as e:
            return report_pb2.ReportResponse(
                success=False,
                message=f"Failed to generate report: {str(e)}"
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    report_pb2_grpc.add_ReportServiceServicer_to_server(ReportService(), server)
    server.add_insecure_port('[::]:50051')
    print("Report gRPC server running on port 50051...")
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Shutting down gRPC server...")
        server.stop(0)

if __name__ == "__main__":
    serve()
