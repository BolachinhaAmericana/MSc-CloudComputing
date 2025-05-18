from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, MapType, BinaryType, ArrayType, IntegerType
import io
from google.cloud import storage
import pydicom
from pydicom.dataset import FileMetaDataset
import numpy as np
import os 
from PIL import Image 

# New top-level function for worker-side GCS interaction and DICOM processing
def _fetch_and_process_dicom_from_gcs_worker(gcs_path_str: str):
    """
    Worker function to download DICOM bytes from GCS and process them.
    A new GCS client is created within this function for worker-side execution.
    """
    try:
        # Create GCS client inside the UDF/worker task
        gcs_client = storage.Client()
        
        if not gcs_path_str.startswith("gs://"):
            raise ValueError(f"Invalid GCS path format: {gcs_path_str}. Must start with gs://")
        
        path_without_scheme = gcs_path_str[5:]
        try:
            bucket_name, blob_name = path_without_scheme.split("/", 1)
        except ValueError:
            raise ValueError(f"Invalid GCS path format: {gcs_path_str}. Expected gs://bucket/object.")

        if not bucket_name or not blob_name:
            raise ValueError(f"Invalid GCS path format: {gcs_path_str}. Bucket or blob name is missing.")

        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        dicom_bytes_content = blob.download_as_bytes()
        
        # Call the static DICOM processing method from SparkDicomProcessor class
        return SparkDicomProcessor.process_dicom_bytes(dicom_bytes_content)
    except Exception as e:
        # Return a structured error compatible with the UDF's output_schema
        error_metadata = {
            "error": f"GCS/Worker processing error for {gcs_path_str}: {str(e)}",
            "patient_id": "", 
            "patient_name": "", 
            "obs": ""
        }
        return (error_metadata, None, None)

class SparkDicomProcessor:
    def __init__(self, app_name="DICOM Processing from GCS with Spark", gcs_bucket_name=None, gcs_prefix=""):
        self.spark = SparkSession.builder.appName(app_name).getOrCreate()
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_prefix = gcs_prefix
        # This storage_client is for driver-side operations like listing files
        self.storage_client = storage.Client()

    @staticmethod
    def process_dicom_bytes(dicom_bytes_content):
        try:
            dcm = pydicom.dcmread(io.BytesIO(dicom_bytes_content), force=True)
            metadata_dict = {}
            for elem in dcm:
                if elem.VR != 'SQ':
                    try:
                        metadata_dict[str(elem.keyword)] = str(elem.value)
                    except Exception:
                        metadata_dict[str(elem.keyword)] = "Error reading value"
            
            patient_id = metadata_dict.get('PatientID', '')
            patient_name = metadata_dict.get('PatientName', '')
            series_description = metadata_dict.get('SeriesDescription', '')

            extracted_metadata = {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "obs": series_description,
            }

            if not hasattr(dcm, 'file_meta') or dcm.file_meta is None:
                dcm.file_meta = FileMetaDataset()
            if not hasattr(dcm.file_meta, 'TransferSyntaxUID') or not dcm.file_meta.TransferSyntaxUID:
                dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            
            pixel_array = dcm.pixel_array # pydicom handles PixelRepresentation to give correct dtype
            image_data_bytes = pixel_array.tobytes()
            image_shape = list(pixel_array.shape)

            return (extracted_metadata, image_data_bytes, image_shape)
        except Exception as e:
            return ({"error": str(e), "patient_id": "", "patient_name": "", "obs": ""}, None, None)

    def _list_dicom_files_gcs(self):
        if not self.gcs_bucket_name:
            raise ValueError("GCS bucket name must be provided.")
        
        blobs = self.storage_client.list_blobs(self.gcs_bucket_name, prefix=self.gcs_prefix)
        dicom_file_gcs_paths = []
        for blob in blobs:
            if blob.name.lower().endswith(".dcm"):
                dicom_file_gcs_paths.append(f"gs://{self.gcs_bucket_name}/{blob.name}")
        return dicom_file_gcs_paths

    def process_dicoms(self):
        dicom_file_gcs_paths = self._list_dicom_files_gcs()

        if not dicom_file_gcs_paths:
            print(f"No .dcm files found in gs://{self.gcs_bucket_name}/{self.gcs_prefix}")
            return None

        paths_df = self.spark.createDataFrame([(path,) for path in dicom_file_gcs_paths], ["gcs_path"])

        output_schema = StructType([
            StructField("metadata", MapType(StringType(), StringType()), True),
            StructField("image_data_bytes", BinaryType(), True),
            StructField("image_shape", ArrayType(IntegerType()), True)
        ])
        
        process_dicom_udf = udf(_fetch_and_process_dicom_from_gcs_worker, output_schema)

        processed_dicoms_df = paths_df.withColumn("processed_data", process_dicom_udf(col("gcs_path")))

        final_df = processed_dicoms_df.select(
            col("gcs_path"),
            col("processed_data.metadata").alias("metadata"),
            col("processed_data.image_data_bytes").alias("image_data_bytes"),
            col("processed_data.image_shape").alias("image_shape")
        )
        return final_df

    def stop_spark(self):
        self.spark.stop()


def main():
    gcs_bucket_name = "msc-g21-dcm_data"
    gcs_prefix = "" 

    output_dir = "processed_dicom_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: {os.path.abspath(output_dir)}")

    processor = SparkDicomProcessor(gcs_bucket_name=gcs_bucket_name, gcs_prefix=gcs_prefix)
    
    final_df = processor.process_dicoms()



    if final_df:
        print("Processing complete. Collecting results to driver for verification...")
        collected_data = final_df.collect()
        
        print(f"\n--- Verification of {len(collected_data)} DICOM files ---")
        for i, row in enumerate(collected_data):
            print(f"\n--- File {i+1}: {row.gcs_path} ---")
            
            print("Metadata:")
            if row.metadata:
                for key, value in row.metadata.items():
                    print(f"  {key}: {value}")
            else:
                print("  No metadata extracted or error in metadata processing.")

            base_filename = os.path.basename(row.gcs_path).replace(".dcm", "")
            
            if row.image_data_bytes and row.image_shape:
                print("Image Data:")
                print(f"  Shape: {row.image_shape}")
                print(f"  Data bytes length: {len(row.image_data_bytes)}")

                try:
                    # Corrected dtype based on your DICOM generation (unsigned 16-bit)
                    correct_dtype = np.uint16 
                    numpy_array = np.frombuffer(row.image_data_bytes, dtype=correct_dtype).reshape(row.image_shape)
                    print(f"  Successfully reconstructed NumPy array with dtype: {correct_dtype}")

                    npy_filename = os.path.join(output_dir, f"{base_filename}.npy")
                    np.save(npy_filename, numpy_array)
                    print(f"  Raw image array saved to: {npy_filename}")

                    image_to_visualize = numpy_array
                    if numpy_array.ndim == 3 and len(row.image_shape) == 3: 
                        image_to_visualize = numpy_array[0] 
                        print(f"  Saving first frame of shape {image_to_visualize.shape} as PNG.")
                    elif numpy_array.ndim != 2:
                        print(f"  Skipping PNG save: array dimension {numpy_array.ndim} not handled for simple PNG conversion.")
                        continue # Corrected from 'pass' to 'continue'
                    
                    if image_to_visualize.size == 0:
                        print("  Skipping PNG save: image array is empty.")
                        continue # Corrected from 'pass' to 'continue'

                    min_val = np.min(image_to_visualize)
                    max_val = np.max(image_to_visualize)
                    
                    if max_val == min_val: 
                        normalized_array = np.zeros_like(image_to_visualize, dtype=np.uint8)
                    else:
                        normalized_array = ((image_to_visualize - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                    
                    pil_image = Image.fromarray(normalized_array)
                    png_filename = os.path.join(output_dir, f"{base_filename}.png")
                    pil_image.save(png_filename)
                    print(f"  Image visualized and saved to: {png_filename}")

                except Exception as e:
                    print(f"  Error processing/saving image for {row.gcs_path}: {e}")
            elif row.metadata and "error" in row.metadata and row.metadata["error"]:
                 print(f"  Skipping image processing due to previous error: {row.metadata['error']}")
            else:
                print("  No image data found or error in image processing.")
        print("\n--- Verification Complete ---")
            
    processor.stop_spark()

if __name__ == "__main__":
    main()
