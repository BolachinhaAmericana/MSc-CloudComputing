from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, MapType, BinaryType, ArrayType, IntegerType, FloatType, DoubleType

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import io
import tempfile
import os
import matplotlib.pyplot as plt

from google.cloud import storage
import google.auth
from typing import Tuple

# Worker-side function to perform inference on a single image
def perform_inference_worker(tensor_bytes) -> Tuple[str, float]:
    """
    Worker function to perform inference on a preprocessed image tensor.
    This loads the model and runs inference on the worker node.
    """
    if tensor_bytes is None:
        return ("ERROR", 0.0)
    
    try:
        # Recreate tensor from bytes
        tensor_array = np.frombuffer(tensor_bytes, dtype=np.float32)
        
        # Reshape to match model input (1, 1, 224, 224)
        # Assuming tensor is already normalized and processed correctly
        tensor = torch.from_numpy(tensor_array).reshape(1, 1, 224, 224)
        
        # Load model - we need to do this inside the worker function
        # since we can't broadcast the model directly
        inference_handler = XrayInferenceHandler(
            bucket_name="pneumonia-models", 
            model_blob_path="modelFINAL02.pth"
        )
        
        # Run inference
        predicted_class, confidence = inference_handler.predict(tensor)
        
        return (predicted_class, float(confidence))
    
    except Exception as e:
        return (f"Error: {str(e)}", 0.0)

class XrayInferenceHandler:
    """
    Handler for loading the model and performing inference.
    Based on the original InferenceHandler from the default services.
    """
    def __init__(self, bucket_name="pneumonia-models", model_blob_path="modelFINAL02.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(bucket_name, model_blob_path)
        self.model.eval()
        
    def _load_model(self, bucket_name, model_blob_path):
        """Load the model from Google Cloud Storage"""
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(model_blob_path)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as temp_file:
                blob.download_to_file(temp_file)
                temp_file_path = temp_file.name
                
            # Initialize the base model using torchxrayvision
            model = xrv.models.DenseNet(weights="densenet121-res224-rsna", op_threshs=None)
            model.eval()
            num_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_features, 2)
            model.op_threshs = None
            model.to(self.device)
            
            # Load trained weights and handle key mismatches
            state_dict = torch.load(temp_file_path, map_location=self.device)
            new_state_dict = {}
            
            for k, v in state_dict.items():
                if k.startswith("classifier.1."):
                    new_k = k.replace("classifier.1.", "classifier.")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
            
            # Clean up temp file
            os.remove(temp_file_path)
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Fallback to a basic model if loading fails
            model = xrv.models.DenseNet(weights="densenet121-res224-rsna")
            model.eval()
            num_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_features, 2)
            model.to(self.device)
            return model
    
    def predict(self, processed_tensor: torch.Tensor) -> Tuple[str, float]:
        """Predict the class of the input image tensor."""
        processed_tensor = processed_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(processed_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        if predicted_class.item() == 0:
            predicted_class = "NORMAL"
        else:
            predicted_class = "PNEUMONIA"
        
        return predicted_class, (confidence.item() * 100)  # e.g., ("NORMAL", 95.0)

class SparkInferenceService:
    """
    Service class for running inference on preprocessed images using Spark.
    """
    def __init__(self,spark, app_name="X-Ray Inference with Spark"):
        """Initialize Spark session and configuration"""
        # self.spark = (SparkSession.builder
        #              .appName(app_name)
        #              .config("spark.executor.memory", "4g") 
        #              .config("spark.driver.memory", "2g")
        #              .config("spark.python.worker.memory", "2g")
        #              .config("spark.executor.cores", "2")
        #              .getOrCreate())
        self.spark = spark

        self.spark.sparkContext.setLogLevel("ERROR")
    
    def run_inference(self, preprocessed_df):
        """
        Run inference on a dataframe of preprocessed images.
        
        Args:
            preprocessed_df: DataFrame from SparkImagePreprocessor with processed tensors
            
        Returns:
            DataFrame with original data plus inference results
        """

        """ Old version """

        if preprocessed_df is None:
            print("No preprocessed dataframe provided")
            return None
        
        # Define schema for inference results
        inference_schema = StructType([
            StructField("predicted_class", StringType(), True),
            StructField("confidence", DoubleType(), True)
        ])
        
        # Create UDF for inference
        inference_udf = udf(perform_inference_worker, inference_schema)
        
        # Apply inference to each row
        print("Running inference on preprocessed images...")
        inference_df = preprocessed_df.withColumn(
            "inference_result",
            inference_udf(col("processed_tensor"))
        )
        
        # Expand inference results into separate columns
        result_df = inference_df.select(
            col("gcs_path"),
            col("metadata"),
            col("image_data_bytes"),
            col("image_shape"),
            col("inference_result.predicted_class").alias("prediction"), # Only nuclear to send
            col("inference_result.confidence").alias("confidence") # Only nuclear to send
        )
        
        print(f"Completed inference on {result_df.count()} images")
        return result_df

       

        
    
    def display_inference_results(self, inference_df, num_samples=2):
        """
        Display sample images with their inference results
        """
        if inference_df is None:
            print("No inference dataframe provided")
            return
            
        # Collect sample rows to driver
        sample_rows = inference_df.limit(num_samples).collect()
        
        if not sample_rows:
            print("No valid processed images found to display")
            return
            
        for i, row in enumerate(sample_rows):
            try:
                # Get processed image data
                png_bytes = row["processed_png"]
                path = row["gcs_path"]
                prediction = row["prediction"]
                confidence = row["confidence"]
                patient_id = row["metadata"].get("patient_id", "Unknown")
                
                # Create a figure
                plt.figure(figsize=(8, 10))
                
                # Display processed image
                png_io = io.BytesIO(png_bytes)
                processed_img = plt.imread(png_io)
                plt.imshow(processed_img, cmap='gray')
                plt.axis('off')
                
                # Add text with prediction results
                result_text = f"Prediction: {prediction}\nConfidence: {confidence:.1f}%"
                plt.text(10, 20, result_text, color='white', fontsize=12, 
                         bbox=dict(facecolor='black', alpha=0.7))
                
                plt.title(f"Patient ID: {patient_id}", fontsize=14)
                plt.tight_layout()
                plt.show()
                
                print(f"Sample {i+1}: {path} - {prediction} ({confidence:.1f}%)")
                
            except Exception as e:
                print(f"Error displaying inference result: {str(e)}")
    
    def stop_spark(self):
        """Stop the Spark session"""
        self.spark.stop()

def main():
    """Main function to demonstrate the complete pipeline"""
    # Step 1: Load and process DICOM files
    from dcm import SparkDicomProcessor
    from preprocessing import SparkImagePreprocessor
    
    gcs_bucket_name = "msc-g21-dcm_data"
    gcs_prefix = ""
    
    # Step 1: DICOM processing
    dicom_processor = SparkDicomProcessor(
        gcs_bucket_name=gcs_bucket_name, 
        gcs_prefix=gcs_prefix
    )
    dicom_df = dicom_processor.process_dicoms()
    
    if dicom_df is None:
        print("No DICOM files were processed")
        dicom_processor.stop_spark()
        return
    print("##########################")
    print("##########################")
    print("##########################")
    print("##########################")
    print("##########################")
    print("##########################")
    # Step 2: Preprocess images
    preprocessor = SparkImagePreprocessor()
    preprocessed_df = preprocessor.preprocess_images(dicom_df)
    
    if preprocessed_df is None:
        print("Image preprocessing failed")
        dicom_processor.stop_spark()
        return
    print("##########################")
    print("##########################")
    print("##########################")
    print("##########################")
    print("##########################")
    print("##########################")
    print("##########################")
    # Step 3: Run inference
    inference_service = SparkInferenceService()
    inference_df = inference_service.run_inference(preprocessed_df)
    
    # Step 4: Display and analyze results
    if inference_df:
        # Show prediction statistics
        inference_df.groupBy("prediction").count().show()
        
        # Show confidence statistics
        inference_df.select("prediction", "confidence").describe().show()
        
        # Show sample of results with all information
        print("\nSample of inference results:")
        inference_df.select(
            col("gcs_path"),
            col("metadata.patient_id"), 
            col("prediction"),
            col("confidence")
        ).show(5, truncate=50)
        
        # Display sample images with predictions
        print("\nDisplaying sample predictions...")
        inference_service.display_inference_results(inference_df, num_samples=2)
    
    # Clean up
    inference_service.stop_spark()

if __name__ == "__main__":
    main()
