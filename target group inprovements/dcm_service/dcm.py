from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, MapType, BinaryType, ArrayType, IntegerType
import io
from google.cloud import storage
from google.oauth2 import service_account
import pydicom
from pydicom.dataset import FileMetaDataset
import numpy as np
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# New top-level function for worker-side GCS interaction and DICOM processing
# Define path to credentials file
# CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./secrets.json")
CREDENTIALS_PATH = os.path.expanduser(os.path.expandvars(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./secrets.json")))
# Function to create a GCS client with credentials
def create_authenticated_storage_client():
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return storage.Client(credentials=credentials)

# New top-level function for worker-side GCS interaction and DICOM processing
def _fetch_and_process_dicom_from_gcs_worker(gcs_path_str: str):
    """
    Worker function to download DICOM bytes from GCS and process them.
    A new GCS client is created within this function for worker-side execution.
    """
    try:
        # Create GCS client inside the UDF/worker task with credentials
        gcs_client = create_authenticated_storage_client()
        
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
    def __init__(self, spark, app_name="DICOM Processing from GCS with Spark", gcs_bucket_name=None, gcs_prefix="", credentials_path=None, batch_size=2, max_images=4):
        """
        Initialize SparkDicomProcessor with batching and capping parameters.
        
        Args:
            spark: SparkSession object
            app_name: Name of the Spark application
            gcs_bucket_name: Name of the GCS bucket
            gcs_prefix: Prefix for file listing in GCS
            credentials_path: Path to GCS credentials file
            batch_size: Number of images to process per batch (default: 50)
            max_images: Maximum number of images to process (default: 640)
        """
        self.spark = spark
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_prefix = gcs_prefix
        self.batch_size = batch_size
        self.max_images = max_images
        
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # Use provided credentials path or default
        self.credentials_path = credentials_path or CREDENTIALS_PATH
        
        # Broadcast credentials content to workers
        with open(self.credentials_path, 'r') as f:
            credentials_content = f.read()
        self.credentials_content = self.spark.sparkContext.broadcast(credentials_content)
        
        # Storage client for driver-side operations
        credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
        self.storage_client = storage.Client(credentials=credentials)

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
            
            pixel_array = dcm.pixel_array
            image_data_bytes = pixel_array.tobytes()
            image_shape = list(pixel_array.shape)
            
            return (extracted_metadata, image_data_bytes, image_shape)
        except Exception as e:
            return ({"error": str(e), "patient_id": "", "patient_name": "", "obs": ""}, None, None)

    def list_dicom_files(self, max_images=None):
        """
        List DICOM files in GCS bucket, up to max_images.
        
        Args:
            max_images: Maximum number of .dcm files to return (optional)
        
        Returns:
            List of GCS paths to .dcm files
        """
        if not self.gcs_bucket_name:
            raise ValueError("GCS bucket name must be provided.")
        
        max_images = max_images if max_images is not None else self.max_images
        blobs = self.storage_client.list_blobs(self.gcs_bucket_name, prefix=self.gcs_prefix)
        dicom_file_gcs_paths = []
        for blob in blobs:
            if blob.name.lower().endswith(".dcm"):
                dicom_file_gcs_paths.append(f"gs://{self.gcs_bucket_name}/{blob.name}")
                if max_images and len(dicom_file_gcs_paths) >= max_images:
                    break
        return dicom_file_gcs_paths

    def process_file_paths(self, file_paths):
        """
        Process a list of DICOM file paths and return a DataFrame.
        Adds patient_id as a top-level column for easier downstream use.
        """
        if not file_paths:
            print("No file paths provided to process.")
            return None
        
        paths_df = self.spark.createDataFrame([(path,) for path in file_paths], ["gcs_path"])
        
        output_schema = StructType([
            StructField("metadata", MapType(StringType(), StringType()), True),
            StructField("image_data_bytes", BinaryType(), True),
            StructField("image_shape", ArrayType(IntegerType()), True)
        ])
        
        process_dicom_udf = udf(_fetch_and_process_dicom_from_gcs_worker, output_schema)
        
        processed_dicoms_df = paths_df.withColumn("processed_data", process_dicom_udf(col("gcs_path")))
        
        # Add patient_id as a top-level column for easier access
        final_df = processed_dicoms_df.select(
            col("gcs_path"),
            col("processed_data.metadata").alias("metadata"),
            col("processed_data.image_data_bytes").alias("image_data_bytes"),
            col("processed_data.image_shape").alias("image_shape"),
            col("processed_data.metadata")["patient_id"].alias("patient_id")
        )
        return final_df

    def process_dicoms(self):
        """
        Process all DICOM files up to max_images (for backward compatibility).
        
        Returns:
            Spark DataFrame with processed DICOM data
        """
        dicom_file_gcs_paths = self.list_dicom_files()
        
        if not dicom_file_gcs_paths:
            print(f"No .dcm files found in gs://{self.gcs_bucket_name}/{self.gcs_prefix}")
            return None
        
        return self.process_file_paths(dicom_file_gcs_paths)

    def stop_spark(self):
        self.spark.stop()

def main_dcm():
    gcs_bucket_name = "msc-g21-dcm_data"
    gcs_prefix = ""  # Optional: "dicom_data/" 
    gcs_output_path = f"gs://dcm_output/processed_dicoms/"
    app_name = "Test_DCM"

    temp_spark = (SparkSession.builder
                 .appName(app_name)
                 .config("spark.driver.memory", "12g")  # Adjusted for 16 GB RAM
                 .config("spark.executor.memory", "2g")  # Relevant for GKE, not local[*]
                 .config("spark.python.worker.memory", "512m")  # Reduced to minimize Python overhead
                 .config("spark.executor.cores", "2")
                 .config("spark.sql.execution.arrow.pyspark.enabled", "true")  # Correct syntax
                 .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")  # Fallback if Arrow fails
                 .config("spark.memory.fraction", "0.8")
                 .config("spark.memory.storageFraction", "0.3")  # Balance storage vs. execution
                     .getOrCreate())
 

    processor = SparkDicomProcessor(spark=temp_spark,gcs_bucket_name=gcs_bucket_name, gcs_prefix=gcs_prefix)
    
    final_df = processor.process_dicoms()

    
    output_gcs_path = f"{gcs_output_path}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.info(f"Writing processed DICOM data to {output_gcs_path}")
    try:
        final_df.write.mode("overwrite").parquet(output_gcs_path)
        logger.info("DICOM data saved successfully to GCS.")
    except Exception as e:
        logger.error(f"Failed to save DICOM data to GCS: {e}")
        
    processor.stop_spark()

    # if final_df:
    #     print("Showing processed DICOM data (first 5 rows):")
    #     final_df.show(n=5, truncate=50)

    # print("Spark configuration:")
    # for k, v in processor.spark.sparkContext.getConf().getAll():
    #     if 'memory' in k or 'cores' in k:
    #         print(f"{k}: {v}")
    print("Stopping Spark")
    processor.stop_spark()

# if __name__ == "__main__":
#     main_dcm()