import os
import numpy as np
import torch
from torchvision.transforms import v2 as vision_transforms 
import torch.nn.functional as F 
import torchxrayvision as xrv 
from google.cloud import storage

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import (
    StructType, StructField, StringType, MapType, BinaryType,
    ArrayType, IntegerType, FloatType, Row
)

# --- 3. Inference Service (based on inference.py) ---

_MODEL_INSTANCE_CACHE = {} # Cache for model instances per GCS path
MODEL_PATH_GCS = "gs://pneumonia-models/modelFINAL02.pth" 
GCS_PROJECT_ID = "cloud-computing-2025" 
def _get_model_instance_worker(model_gcs_path_str= MODEL_PATH_GCS ,worker_pid):
    """
    Loads the model. Called per worker process if model not in cache.
    Downloads from GCS if path starts with 'gs://'.
    """
    cache_key = (model_gcs_path_str, worker_pid)
    if cache_key in _MODEL_INSTANCE_CACHE:
        return _MODEL_INSTANCE_CACHE[cache_key]

    print(f"Worker (PID {worker_pid}): Loading model from {model_gcs_path_str}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    local_model_path = model_gcs_path_str
    if model_gcs_path_str.startswith("gs://"):
        try:
            client = storage.Client(project=GCS_PROJECT_ID) # Ensure project is used
            _, blob_name = model_gcs_path_str.replace("gs://", "").split("/", 1)
            # Bucket name is part of model_gcs_path_str, e.g., gs://<bucket-name>/path/to/model
            bucket_name = model_gcs_path_str.split("/")[2]
            
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Create a unique temporary path for the model file
            temp_dir = f"/tmp/spark_models_{worker_pid}"
            os.makedirs(temp_dir, exist_ok=True)
            local_model_path = os.path.join(temp_dir, os.path.basename(blob_name))
            
            blob.download_to_filename(local_model_path)
            print(f"Worker (PID {worker_pid}): Model downloaded to {local_model_path} for GCS path {model_gcs_path_str}")
        except Exception as e:
            raise RuntimeError(f"Worker (PID {worker_pid}): Failed to download model from GCS {model_gcs_path_str}: {e}")

    # Load model (logic from original inference.py)
    # Base model from torchxrayvision
    model = xrv.models.DenseNet(weights="densenet121-res224-rsna", op_threshs=None) 
    num_features = model.classifier.in_features
    # Adjust classifier for 2 classes (NORMAL, PNEUMONIA)
    model.classifier = torch.nn.Linear(num_features, 2) 
    
    try:
        state_dict = torch.load(local_model_path, map_location=device)
        # Handle potential key mismatches if model was saved with DataParallel or different naming
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if k.startswith("module."): # If saved with DataParallel
                name = k[7:]
            if name.startswith("classifier.1."): # Specific to an older torchxrayvision format
                name = name.replace("classifier.1.", "classifier.")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except Exception as e:
         raise RuntimeError(f"Worker (PID {worker_pid}): Failed to load model state_dict from {local_model_path}: {e}")

    model.to(device)
    model.eval() # Set to evaluation mode
    
    print(f"Worker (PID {worker_pid}): Model loaded successfully on device {device} from {model_gcs_path_str}")
    _MODEL_INSTANCE_CACHE[cache_key] = (model, device)
    return model, device

def _infer_image_worker(tensor_bytes, tensor_shape_list, model_gcs_path_for_udf):
    try:
        if tensor_bytes is None or not tensor_shape_list:
            return (None, None, "Missing tensor data or shape for inference")

        worker_pid = os.getpid() # Get current process ID for model caching
        model, device = _get_model_instance_worker(model_gcs_path_for_udf, worker_pid)
        
        tensor_shape = tuple(tensor_shape_list) # E.g., [1, 224, 224]
        # Tensor expected to be float32 from preprocessing
        processed_tensor = torch.from_numpy(
            np.frombuffer(tensor_bytes, dtype=np.float32).reshape(tensor_shape)
        )
        processed_tensor = processed_tensor.to(device)

        # Model expects batch dimension [B, C, H, W]. Preprocessing outputs [C, H, W].
        if processed_tensor.ndim == 3:
             processed_tensor = processed_tensor.unsqueeze(0) # Add batch dimension

        with torch.no_grad(): # Disable gradient calculations for inference
            output = model(processed_tensor) # Output shape: [B, num_classes]
            probabilities = F.softmax(output, dim=1) # Apply softmax to get probabilities
            confidence_tensor, predicted_class_tensor = torch.max(probabilities, dim=1) # Get max prob and its index
        
        predicted_class_idx = predicted_class_tensor.item() # Convert tensor to Python number
        confidence_score = confidence_tensor.item() * 100.0 # Convert to percentage

        # Assuming 0 for NORMAL, 1 for PNEUMONIA (adjust if your model's classes are different)
        prediction_label = "NORMAL" if predicted_class_idx == 0 else "PNEUMONIA"
        
        return (prediction_label, confidence_score, None) # No error
    except Exception as e:
        # import traceback
        # tb_str = traceback.format_exc()
        return (None, None, f"Inference error: {str(e)}")


class SparkInferenceService:
    def __init__(self, spark_session, model_gcs_path_param):
        self.spark = spark_session
        self.model_gcs_path = model_gcs_path_param

    def process(self, input_df):
        if input_df is None:
            print("InferenceService: Input DataFrame is None.")
            return None
        
        if not self.model_gcs_path or self.model_gcs_path == "gs://your-model-bucket/path/to/your_model.pth":
             raise ValueError("Model GCS path is not configured correctly for InferenceService.")


        schema = StructType([
            StructField("prediction", StringType(), True),
            StructField("confidence", FloatType(), True),
            StructField("error_message", StringType(), True)
        ])
        
        # Pass the model GCS path to the UDF.
        # Each worker will use this path to load the model via _get_model_instance_worker.
        inference_udf = udf(lambda tb, ts: _infer_image_worker(tb, ts, self.model_gcs_path), schema)

        inferred_df = input_df.withColumn("inference_result", inference_udf(
            col("tensor_bytes"), col("tensor_shape")
        ))
        
        final_df = inferred_df.select(
            input_df["*"], # Keep all columns from the input DataFrame
            col("inference_result.prediction").alias("prediction"),
            col("inference_result.confidence").alias("confidence"),
            col("inference_result.error_message").alias("inference_error")
        ).filter(col("prediction").isNotNull()) # Filter out rows where inference itself failed
        return final_df
    
    
    
# Pode ter erros, fazer debugging
def main_inference_service_test():
    spark = SparkSession.builder.appName("InferenceServiceTest").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("--- Testing SparkInferenceService ---")
    print(f"Using Model Path: {MODEL_PATH_GCS}")
    if MODEL_PATH_GCS == "gs://your-model-bucket/path/to/your_model.pth":
        print("WARNING: MODEL_PATH_GCS is set to the default placeholder. Inference might fail if a valid model is not at this path.")

    mock_tensor_array = np.random.rand(1, 224, 224).astype(np.float32)
    mock_tensor_bytes = mock_tensor_array.tobytes()
    mock_tensor_shape = list(mock_tensor_array.shape)

    mock_preprocessing_data = [
        Row(
            gcs_path="gs://mock-bucket/test1.dcm", 
            metadata={"patient_id": "TestPatient001", "patient_name": "John Doe", "error": None}, 
            image_data_bytes=b"original_image_bytes1", 
            image_shape=[32,32], 
            image_dtype_str='uint16', 
            tensor_bytes=mock_tensor_bytes,
            tensor_shape=mock_tensor_shape,
            preprocessing_error=None
        )
    ]
    mock_preprocessing_schema = StructType([
        StructField("gcs_path", StringType(), True),
        StructField("metadata", MapType(StringType(), StringType()), True),
        StructField("image_data_bytes", BinaryType(), True),
        StructField("image_shape", ArrayType(IntegerType()), True),
        StructField("image_dtype_str", StringType(), True),
        StructField("tensor_bytes", BinaryType(), True),
        StructField("tensor_shape", ArrayType(IntegerType()), True),
        StructField("preprocessing_error", StringType(), True)
    ])
    mock_preprocessing_output_df = spark.createDataFrame(mock_preprocessing_data, schema=mock_preprocessing_schema)
    print("Mock input to InferenceService:")
    mock_preprocessing_output_df.select("gcs_path", "metadata.patient_id", "tensor_shape").show()


    inference_service = SparkInferenceService(spark, MODEL_PATH_GCS)
    inferred_output_df = inference_service.process(mock_preprocessing_output_df)

    if inferred_output_df:
        print(f"Inference Service processed {inferred_output_df.count()} items.")
        inferred_output_df.printSchema()
        inferred_output_df.select("gcs_path", "metadata.patient_id", "prediction", "confidence", "inference_error").show(truncate=False)
    else:
        print("Inference Service did not produce any output.")
        
        
        
        
    
print("\nTesting Inference Service...")
main_inference_service_test() # Make sure MODEL_PATH_GCS is valid 

    