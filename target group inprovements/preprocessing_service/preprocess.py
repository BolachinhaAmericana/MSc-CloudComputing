from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, MapType, BinaryType, ArrayType, IntegerType, FloatType

import io
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import cv2
import os
from datetime import datetime
import logging
from google.cloud import storage
from google.oauth2 import service_account

# Setup logging
logger = logging.getLogger(__name__)

# Set up credentials path (same as dcm.py)
CREDENTIALS_PATH = os.path.expanduser(os.path.expandvars(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./secrets.json")))

def image_preprocessing(image_data_bytes, image_shape):
    """
    Worker function to preprocess image data from DICOM files
    - Resizes to 224x224
    - Converts to grayscale
    - Applies Gaussian blur
    - Applies histogram equalization
    - Normalizes to prepare for model inference
    """
    try:
        if image_data_bytes is None or image_shape is None:
            return None
        # Reconstruct image from bytes and shape
        img_array = np.frombuffer(image_data_bytes, dtype=np.uint16)
        img_array = img_array.reshape(image_shape)
        # Scale to 0-255 for PIL
        img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        # Create transform pipeline
        transform = transforms.Compose([
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
        # Process the image
        processed_tensor = transform(image)
        # Convert processed tensor to bytes
        tensor_bytes = processed_tensor.numpy().tobytes()
        return tensor_bytes
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

class SparkImagePreprocessor:
    def __init__(self, spark, output_bucket="preprocess_output"):
        self.spark = spark
        self.output_bucket = output_bucket
        self.spark.sparkContext.setLogLevel("ERROR")
        
    def preprocess_images(self, dicom_df):
        """
        Process a dataframe of DICOM images (from Parquet)
        Returns a dataframe with original data + processed tensor
        """
        if dicom_df is None:
            print("No DICOM dataframe provided")
            return None
        # Define schema for the processed image data
        output_schema = BinaryType()
        # Create UDF for preprocessing
        preprocess_udf = udf(
            lambda image_bytes, image_shape: image_preprocessing(image_bytes, image_shape), 
            output_schema
        )
        # Apply preprocessing
        print("Preprocessing DICOM images...")
        processed_df = dicom_df.withColumn(
            "processed_tensor", 
            preprocess_udf(col("image_data_bytes"), col("image_shape"))
        )
        # Select only relevant columns for output
        final_df = processed_df.select(
            col("gcs_path"),
            col("metadata"),
            col("patient_id"),
            col("processed_tensor")
        )
        print(f"Processed {final_df.count()} images")
        return final_df

    def stop_spark(self):
        self.spark.stop()

def main():
    # Set up Spark session
    temp_spark = (
        SparkSession.builder
        .appName("Preprocessing_DICOM")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "4g")
        .config("spark.executor.cores", "2")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    # Path to the latest parquet in dcm_output bucket
    # You may want to automate finding the latest folder, here is a simple static example:
    dcm_output_bucket = "dcm_output"
    dcm_output_prefix = "processed_dicoms"
    
    preprocess_output_bucket = "preprocess_output"
    preprocess_output_prefix = "processed_tensors"
    # Example: gs://dcm_output/processed_dicoms/20240610183000
    # You may want to list and pick the latest, here we just use a placeholder:
    parquet_folder = f"gs://{dcm_output_bucket}/{dcm_output_prefix}/"  # Add timestamp if needed

    # Find the latest folder (optional, simple version)
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    storage_client = storage.Client(credentials=credentials)
    blobs = storage_client.list_blobs(dcm_output_bucket, prefix=dcm_output_prefix + "/")
    folders = set()
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) > 2 and parts[1]:
            folders.add(parts[1])
    if not folders:
        print("No parquet folders found in dcm_output bucket.")
        temp_spark.stop()
        return
    latest_folder = sorted(folders)[-1]
    parquet_folder = f"gs://{dcm_output_bucket}/{dcm_output_prefix}/{latest_folder}"

    print(f"Reading DICOM parquet from: {parquet_folder}")
    dicom_df = temp_spark.read.parquet(parquet_folder)

    # Step 2: Preprocess the images
    preprocessor = SparkImagePreprocessor(spark=temp_spark, output_bucket="preprocess_output")
    processed_df = preprocessor.preprocess_images(dicom_df)
    
    # Save processed DataFrame as Parquet to GCS bucket "preprocess_output"
    output_gcs_path = f"gs://{preprocess_output_bucket}/{preprocess_output_prefix}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.info(f"Writing processed tensors to {output_gcs_path}")
    try:
        processed_df.write.mode("overwrite").parquet(output_gcs_path)
        logger.info("Processed tensors saved successfully to GCS.")
    except Exception as e:
        logger.error(f"Failed to save processed tensors to GCS: {e}")

    preprocessor.stop_spark()

if __name__ == "__main__":
    main()