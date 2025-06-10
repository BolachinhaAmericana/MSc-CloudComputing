from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, MapType, BinaryType, ArrayType, IntegerType, FloatType

import matplotlib.pyplot as plt


import io
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import cv2
from google.cloud import storage
from google.oauth2 import service_account

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
            return (None, None)
            
        # Reconstruct image from bytes and shape
        img_array = np.frombuffer(image_data_bytes, dtype=np.uint16)
        img_array = img_array.reshape(image_shape)
        
        # Scale to 0-255 for PIL
        img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        # Create transform pipeline similar to Phase4.py
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
        
        # Create normalized image for saving
        img_to_save = processed_tensor.squeeze(0)  # Remove channel dim
        img_to_save = (img_to_save * 127.5 + 127.5).byte()  # Denormalize
        processed_pil = Image.fromarray(img_to_save.numpy())
        
        # Convert to PNG bytes
        img_byte_arr = io.BytesIO()
        processed_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        png_bytes = img_byte_arr.getvalue()
        
        return (tensor_bytes, png_bytes)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return (None, None)

def upload_processed_image_to_gcs(png_bytes, gcs_path):
    """
    Upload processed image to GCS
    """
    try:
        if png_bytes is None:
            return False
            
        client = storage.Client()
        bucket_name = gcs_path.split("/")[2]
        blob_name = "/".join(gcs_path.split("/")[3:])
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_string(png_bytes, content_type='image/png')
        return True
    except Exception as e:
        print(f"Error uploading to GCS: {str(e)}")
        return False

class SparkImagePreprocessor:
    def __init__(self,spark, app_name="DICOM Image Preprocessing with Spark", output_bucket="xray-bucket-fcul"):
        """
        Initialize Spark session and configure preprocessing service
        """
        # self.spark = (SparkSession.builder
        #              .appName(app_name)
        #              .config("spark.executor.memory", "4g")
        #              .config("spark.driver.memory", "2g") 
        #              .config("spark.python.worker.memory", "1g")
        #              .config("spark.executor.cores", "2")
        #              .getOrCreate())
        self.spark = spark
                     
        self.output_bucket = output_bucket

        self.spark.sparkContext.setLogLevel("ERROR")
        
    def preprocess_images(self, dicom_df):
        """
        Process a dataframe of DICOM images (from SparkDicomProcessor)
        Returns a dataframe with original data + processed images
        """

        """ Old version """
        if dicom_df is None:
            print("No DICOM dataframe provided")
            return None
            
        # Define schema for the processed image data
        output_schema = StructType([
            StructField("tensor_bytes", BinaryType(), True),
            StructField("png_bytes", BinaryType(), True)
        ])
        
        # Create UDF for preprocessing
        preprocess_udf = udf(
            lambda image_bytes, image_shape: image_preprocessing(image_bytes, image_shape), 
            output_schema
        )
        
        # Apply preprocessing
        print("Preprocessing DICOM images...")
        processed_df = dicom_df.withColumn(
            "processed_image", 
            preprocess_udf(col("image_data_bytes"), col("image_shape"))
        )
        
        # Expand the result columns
        final_df = processed_df.select(
            # col("gcs_path"),
            # col("metadata"),
            # col("image_data_bytes"),
            # col("image_shape"),
            col("processed_image.tensor_bytes").alias("processed_tensor") # only nuclear to send
        )
        
        print(f"Processed {final_df.count()} images")
        return final_df

        

    def display_sample_image(self, processed_df, num_samples=1):
        """
        Display sample images from the processed dataframe
        No GCS operations involved - operates only on driver
        """
        if processed_df is None:
            print("No processed dataframe provided")
            return
            
        # Collect sample rows to driver
        sample_rows = processed_df.filter(
            col("processed_png").isNotNull()
        ).limit(num_samples).collect()
        
        if not sample_rows:
            print("No valid processed images found to display")
            return
            
        for i, row in enumerate(sample_rows):
            try:
                # Get original and processed image data
                original_bytes = row["image_data_bytes"]
                original_shape = row["image_shape"]
                png_bytes = row["processed_png"]
                path = row["gcs_path"]
                
                patient_id = row["metadata"].get("patient_id", "Unknown")
                
                # Create a figure with subplots
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Display original DICOM image
                original_array = np.frombuffer(original_bytes, dtype=np.uint16)
                original_img = original_array.reshape(original_shape)
                # Scale for display
                original_img_display = (original_img / original_img.max() * 255).astype(np.uint8)
                axes[0].imshow(original_img_display, cmap='gray')
                axes[0].set_title(f"Original DICOM\nPatient ID: {patient_id}")
                axes[0].axis('off')
                
                # Display processed image
                png_io = io.BytesIO(png_bytes)
                processed_img = plt.imread(png_io)
                axes[1].imshow(processed_img, cmap='gray')
                axes[1].set_title(f"Processed (224x224)")
                axes[1].axis('off')
                
                plt.suptitle(f"Sample {i+1}: {path}", fontsize=10)
                plt.tight_layout()
                plt.show()
                
                print(f"Displayed processed image for patient ID: {patient_id}")
                
            except Exception as e:
                print(f"Error displaying image: {str(e)}")
    
    def stop_spark(self):
        """Stop the Spark session"""
        self.spark.stop()

def main():
    

    """Main function to demonstrate the preprocessing pipeline"""
    # Step 1: Load DICOM files using SparkDicomProcessor
    from dcm import SparkDicomProcessor
    
    gcs_bucket_name = "msc-g21-dcm_data"
    gcs_prefix = ""  # Optional: "dicom_data/"
    output_bucket = gcs_bucket_name
    
    # Process DICOM files
    dicom_processor = SparkDicomProcessor(
        gcs_bucket_name=gcs_bucket_name, 
        gcs_prefix=gcs_prefix
    )
    dicom_df = dicom_processor.process_dicoms()
    
    if dicom_df is None:
        print("No DICOM files were processed")
        dicom_processor.stop_spark()
        return

    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("######################################################")

    # Step 2: Preprocess the images
    preprocessor = SparkImagePreprocessor(output_bucket=output_bucket)
    processed_df = preprocessor.preprocess_images(dicom_df)
    
    # Display sample processed images
    if processed_df:
        print("\nDisplaying sample processed images...")
        preprocessor.display_sample_image(processed_df, num_samples=2)
    
    # Optional: Uncomment to save to GCS
    # result_df = preprocessor.save_processed_images(processed_df)
    
    # Clean up
    preprocessor.stop_spark()

if __name__ == "__main__":
    main()