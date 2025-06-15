from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, MapType, BinaryType, ArrayType, IntegerType, FloatType, DoubleType

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import io
import tempfile
import os
from datetime import datetime
import logging
import google.auth
from google.cloud import storage
from google.oauth2 import service_account # Ensure this is imported
from typing import Tuple

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up credentials path (consistent with other services)
CREDENTIALS_PATH = os.path.expanduser(os.path.expandvars(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./secrets.json")))

def create_authenticated_storage_client():
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return storage.Client(credentials=credentials)

# Worker-side function to perform inference on a single image
def perform_inference_worker(tensor_bytes) -> Tuple[str, float]:
    """
    Worker function to perform inference on a preprocessed image tensor.
    This loads the model and runs inference on the worker node.
    """
    if tensor_bytes is None:
        logger.warning("Received null tensor_bytes in worker.")
        return ("ERROR_NULL_INPUT", 0.0)
    
    try:
        # Recreate tensor from bytes
        tensor_array = np.frombuffer(tensor_bytes, dtype=np.float32)
        
        # Reshape to match model input (1, 1, 224, 224)
        tensor = torch.from_numpy(tensor_array).reshape(1, 1, 224, 224)
        
        # Load model - this happens inside the worker function
        # Ensure bucket_name and model_blob_path are correct and accessible
        inference_handler = XrayInferenceHandler(
            bucket_name="pneumonia-models", 
            model_blob_path="modelFINAL02.pth" # Make sure this model exists and is accessible
        )
        
        predicted_class, confidence = inference_handler.predict(tensor)
        
        return (predicted_class, float(confidence))
    
    except Exception as e:
        logger.error(f"Error during inference in worker: {str(e)}", exc_info=True)
        return (f"ERROR_INFERENCE_WORKER", 0.0)

class XrayInferenceHandler:
    """
    Handler for loading the model and performing inference.
    """
    _model_cache = None # Class-level cache for the model
    _device_cache = None

    def __init__(self, bucket_name="pneumonia-models", model_blob_path="modelFINAL02.pth"):
        if XrayInferenceHandler._device_cache is None:
            XrayInferenceHandler._device_cache = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = XrayInferenceHandler._device_cache

        if XrayInferenceHandler._model_cache is None:
            logger.info(f"Loading model for the first time on this worker: gs://{bucket_name}/{model_blob_path}")
            XrayInferenceHandler._model_cache = self._load_model(bucket_name, model_blob_path)
        
        self.model = XrayInferenceHandler._model_cache
        if self.model:
            self.model.eval() # Ensure model is in eval mode
        else:
            raise RuntimeError("Failed to load model for inference.")
        
    def _load_model(self, bucket_name, model_blob_path):
        """Load the model from Google Cloud Storage"""
        temp_file_path = None
        try:
            # Use application default credentials or service account from environment for GCS client
            # No explicit credentials needed if GOOGLE_APPLICATION_CREDENTIALS is set for spark-submit
            # or if running on GCP with appropriate service account.
            # For local UDF execution where Spark might not propagate, ensure worker has access.
            # One way is to ensure the secrets.json is available and used by google.auth.default()
            # or explicitly pass credentials if needed, though tricky with UDFs.
            # The simplest for Spark is often relying on the environment.
            credentials, project = google.auth.default()
            client = create_authenticated_storage_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(model_blob_path)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as temp_file:
                blob.download_to_file(temp_file)
                temp_file_path = temp_file.name
                
            model = xrv.models.DenseNet(weights="densenet121-res224-rsna", op_threshs=None)
            num_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_features, 2) # Assuming 2 classes: NORMAL, PNEUMONIA
            model.op_threshs = None # Ensure this is set if you don't use it
            
            state_dict = torch.load(temp_file_path, map_location=self.device)
            new_state_dict = {}
            
            # Handle potential key mismatches if model was saved from DataParallel or with different naming
            for k, v in state_dict.items():
                name = k.replace("module.", "") # remove `module.` prefix if present
                if name.startswith("classifier.1."): # Adjust if your saved model has this structure
                    name = name.replace("classifier.1.", "classifier.")
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            model.to(self.device)
            logger.info(f"Model loaded successfully to device: {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model gs://{bucket_name}/{model_blob_path}: {str(e)}", exc_info=True)
            return None # Return None to indicate failure
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def predict(self, processed_tensor: torch.Tensor) -> Tuple[str, float]:
        """Predict the class of the input image tensor."""
        if self.model is None:
            raise RuntimeError("Model not loaded, cannot predict.")
            
        processed_tensor = processed_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(processed_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1)

        # Map index to class name
        # Ensure this mapping matches your model's training
        if predicted_idx_tensor.item() == 0:
            predicted_class_name = "NORMAL"
        elif predicted_idx_tensor.item() == 1:
            predicted_class_name = "PNEUMONIA"
        else:
            predicted_class_name = "UNKNOWN" # Should not happen for a 2-class model
        
        return predicted_class_name, (confidence_tensor.item() * 100)

class SparkInferenceService:
    """
    Service class for running inference on preprocessed images using Spark.
    """
    def __init__(self, spark_session): # Accepts a Spark session
        self.spark = spark_session
        self.spark.sparkContext.setLogLevel("ERROR")
    
    def run_inference(self, preprocessed_df):
        """
        Run inference on a dataframe of preprocessed images.
        """
        if preprocessed_df is None:
            logger.error("No preprocessed dataframe provided to run_inference.")
            return None
        
        # Define schema for inference results
        inference_schema = StructType([
            StructField("predicted_class", StringType(), True),
            StructField("confidence", DoubleType(), True)
        ])
        
        # Create UDF for inference
        inference_udf = udf(perform_inference_worker, inference_schema)
        
        logger.info("Running inference on preprocessed images...")
        # Apply inference to each row using the 'processed_tensor' column
        inference_df = preprocessed_df.withColumn(
            "inference_result",
            inference_udf(col("processed_tensor")) # Assumes 'processed_tensor' column exists
        )
        
        # Expand inference results into separate columns
        # Carry over necessary identifiers from the preprocessed_df
        result_df = inference_df.select(
            col("gcs_path"),       # Original GCS path of the DICOM
            col("metadata"),       # Original metadata
            col("patient_id"),     # Patient ID from metadata
            col("inference_result.predicted_class").alias("prediction"),
            col("inference_result.confidence").alias("confidence")
        )
        
        count = result_df.count()
        logger.info(f"Completed inference on {count} images")
        return result_df
    
    def stop_spark(self):
        """Stop the Spark session if this service was responsible for it"""
        # Generally, the entity that creates the Spark session should stop it.
        # If main() creates it, main() should stop it.
        # self.spark.stop() # Commented out, let main handle it.
        pass
    



def main():
    """Main function to run the inference pipeline"""
    spark = None
    try:
        spark = (SparkSession.builder
                 .appName("XRay_Inference_Pipeline")
                 .config("spark.driver.memory", "8g")
                 .config("spark.executor.memory", "4g") # Adjust based on model size and data
                 .config("spark.python.worker.memory", "2g") # Memory for Python UDF workers
                 .config("spark.executor.cores", "2")
                 .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                 .getOrCreate())
        logger.info("Spark session created for inference pipeline.")

        # Define GCS paths
        preprocess_output_bucket = "preprocess_output"
        preprocess_output_prefix = "processed_tensors"
        
        inference_output_bucket = "inference_output1"
        inference_output_prefix = "inference_results"

        # Find the latest preprocessed data folder
        # This requires GCS client on the driver
        credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        storage_client = storage.Client(credentials=credentials)
        
        blobs = storage_client.list_blobs(preprocess_output_bucket, prefix=preprocess_output_prefix + "/")
        folders = set()
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) > 2 and parts[1]: # e.g., "processed_tensors/20230101120000/" -> parts[1] is "20230101120000"
                folders.add(parts[1])
        
        if not folders:
            logger.error(f"No preprocessed data folders found in gs://{preprocess_output_bucket}/{preprocess_output_prefix}/")
            return
        
        latest_folder = sorted(list(folders))[-1]
        input_parquet_path = f"gs://{preprocess_output_bucket}/{preprocess_output_prefix}/{latest_folder}"
        logger.info(f"Reading preprocessed data from: {input_parquet_path}")

        # Read preprocessed data
        preprocessed_df = spark.read.parquet(input_parquet_path)

        if preprocessed_df.rdd.isEmpty():
            logger.warning(f"No data found in {input_parquet_path}. Exiting.")
            return

        # Instantiate and run inference service
        inference_service = SparkInferenceService(spark_session=spark)
        inference_results_df = inference_service.run_inference(preprocessed_df)

        if inference_results_df is None or inference_results_df.rdd.isEmpty():
            logger.error("Inference did not produce any results.")
            return

        # Save inference results to GCS
        output_gcs_path = f"gs://{inference_output_bucket}/{inference_output_prefix}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"Writing inference results to {output_gcs_path}")
        
        inference_results_df.write.mode("overwrite").parquet(output_gcs_path)
        logger.info(f"Inference results saved successfully to {output_gcs_path}")

        # Optional: Show some results
        logger.info("Sample of inference results:")
        inference_results_df.select("gcs_path", "patient_id", "prediction", "confidence").show(10, truncate=True)

        inference_results_df.groupBy("prediction").count().show()

    except Exception as e:
        logger.error(f"An error occurred in the main inference pipeline: {e}", exc_info=True)
    finally:
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()

if __name__ == "__main__":
    main()