import cv2 
import torch 
from torchvision.transforms import v2 as vision_transforms 
from PIL import Image
import numpy as np

# --- 2. Preprocessing Service (based on Phase4.py) ---
# Globally defined transforms to ensure they are available to UDFs
_PREPROCESSING_TRANSFORMS = vision_transforms.Compose([
    vision_transforms.Resize((224, 224), antialias=True),
    vision_transforms.Grayscale(num_output_channels=1),
    vision_transforms.Lambda(lambda pil_img: Image.fromarray(cv2.GaussianBlur(np.array(pil_img), (5, 5), 0))),
    vision_transforms.Lambda(lambda pil_img: Image.fromarray(cv2.equalizeHist(np.array(pil_img)))),
    vision_transforms.ToTensor(), # Converts PIL image or numpy.ndarray to tensor
    vision_transforms.ToDtype(torch.float32, scale=True), # Converts to float32 and scales to [0.0, 1.0]
    vision_transforms.Normalize(mean=[0.5], std=[0.5])    # Normalizes to [-1.0, 1.0]
])

def _numpy_dtype_from_string(dtype_str):
    """Helper to convert dtype string to actual numpy dtype."""
    try:
        return np.dtype(dtype_str)
    except TypeError:
        # Fallback for simple cases if np.dtype(str) fails for some reason
        if 'uint16' in dtype_str: return np.uint16
        if 'int16' in dtype_str: return np.int16
        if 'uint8' in dtype_str: return np.uint8
        if 'int8' in dtype_str: return np.int8
        raise ValueError(f"Unsupported or unknown dtype string: {dtype_str}")


def _preprocess_image_worker(image_bytes, image_shape_list, image_dtype_as_str):
    try:
        if image_bytes is None or not image_shape_list or not image_dtype_as_str:
            return (None, None, "Missing image data, shape, or dtype for preprocessing")
            
        image_shape = tuple(image_shape_list) # E.g., (512, 512) or (10, 512, 512)
        numpy_actual_dtype = _numpy_dtype_from_string(image_dtype_as_str)
        
        pixel_array = np.frombuffer(image_bytes, dtype=numpy_actual_dtype).reshape(image_shape)

        # Handle multi-frame (e.g., cine) or multi-channel (e.g. RGB from DICOM) images:
        # For typical X-rays, we want a single 2D grayscale image.
        if pixel_array.ndim == 3:
            if image_shape[0] < 5: # Likely frames or channels, take the first frame/channel
                pixel_array = pixel_array[0] 
            elif image_shape[2] < 5: # Likely (H, W, C), convert to grayscale if C=3 (RGB)
                if image_shape[2] == 3: # Assuming RGB
                     pixel_array = cv2.cvtColor(pixel_array.astype(np.uint8), cv2.COLOR_RGB2GRAY) # Needs uint8
                else: # just take first channel
                     pixel_array = pixel_array[:,:,0]
            else: # Unclear 3D structure for X-ray, attempt first slice
                pixel_array = pixel_array[0]

        if pixel_array.ndim != 2:
            return (None, None, f"Image after initial processing is not 2D, shape: {pixel_array.shape}")

        # Convert to PIL Image. Input to _PREPROCESSING_TRANSFORMS must be PIL.
        # PIL requires specific modes (e.g., 'L' for grayscale).
        # Normalize pixel values to 8-bit if they are 16-bit for standard PIL 'L' mode.
        if numpy_actual_dtype in [np.uint16, np.int16, np.int32, np.uint32, np.float32, np.float64]:
            # More robust scaling to 0-255 for PIL 'L' mode
            min_val, max_val = pixel_array.min(), pixel_array.max()
            if max_val == min_val:
                pixel_array_scaled = np.zeros_like(pixel_array, dtype=np.uint8)
            else:
                pixel_array_scaled = ((pixel_array - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            pil_image = Image.fromarray(pixel_array_scaled, mode='L')
        elif numpy_actual_dtype == np.uint8:
            pil_image = Image.fromarray(pixel_array, mode='L')
        else:
            return (None, None, f"Unsupported numpy dtype for PIL conversion: {pixel_array.dtype}")
            
        processed_tensor = _PREPROCESSING_TRANSFORMS(pil_image) # Expects PIL image
        
        tensor_bytes = processed_tensor.numpy().tobytes()
        tensor_shape = list(processed_tensor.shape) # Should be [1, 224, 224]
        return (tensor_bytes, tensor_shape, None) # No error
    except Exception as e:
        # import traceback # For detailed debugging if needed
        # tb_str = traceback.format_exc()
        return (None, None, f"Preprocessing error: {str(e)}")


class SparkPreprocessingService:
    def __init__(self, spark_session):
        self.spark = spark_session

    def process(self, input_df):
        if input_df is None:
            print("PreprocessingService: Input DataFrame is None.")
            return None

        schema = StructType([
            StructField("tensor_bytes", BinaryType(), True),
            StructField("tensor_shape", ArrayType(IntegerType()), True),
            StructField("error_message", StringType(), True)
        ])
        preprocess_udf = udf(_preprocess_image_worker, schema)

        processed_df = input_df.withColumn("preprocessing_result", preprocess_udf(
            col("image_data_bytes"), col("image_shape"), col("image_dtype_str")
        ))
        
        final_df = processed_df.select(
            input_df["*"], # Keep all columns from the input DataFrame
            col("preprocessing_result.tensor_bytes").alias("tensor_bytes"),
            col("preprocessing_result.tensor_shape").alias("tensor_shape"),
            col("preprocessing_result.error_message").alias("preprocessing_error")
        ).filter(col("tensor_bytes").isNotNull()) # Filter out rows where preprocessing itself failed critically
        return final_df


# Pode ter erros, fazer debugging
def main_preprocessing_service_test():
    spark = SparkSession.builder.appName("PreprocessingServiceTest").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("--- Testing SparkPreprocessingService ---")

    mock_image_array = np.random.randint(0, 2000, size=(32, 32), dtype=np.uint16)
    mock_image_bytes = mock_image_array.tobytes()
    mock_image_shape = list(mock_image_array.shape)
    mock_image_dtype = 'uint16'

    mock_dcm_data = [
        Row(
            gcs_path="gs://mock-bucket/test1.dcm",
            metadata={"patient_id": "TestPatient001", "patient_name": "John Doe", "error": None}, # Added error field
            image_data_bytes=mock_image_bytes,
            image_shape=mock_image_shape,
            image_dtype_str=mock_image_dtype
        ),
        Row(
            gcs_path="gs://mock-bucket/error_test.dcm",
            metadata={"patient_id": "ErrorPatient003", "patient_name": "Error Prone", "error": None}, # Added error field
            image_data_bytes=mock_image_bytes,
            image_shape=mock_image_shape,
            image_dtype_str='invalid_dtype' 
        )
    ]
    # Schema now matches output of OriginalSparkDicomProcessor
    mock_dcm_schema = StructType([
        StructField("gcs_path", StringType(), True),
        StructField("metadata", MapType(StringType(), StringType()), True),
        StructField("image_data_bytes", BinaryType(), True),
        StructField("image_shape", ArrayType(IntegerType()), True),
        StructField("image_dtype_str", StringType(), True)
    ])
    mock_dcm_output_df = spark.createDataFrame(mock_dcm_data, schema=mock_dcm_schema)
    print("Mock input to PreprocessingService:")
    mock_dcm_output_df.show(truncate=False)

    preprocess_service = SparkPreprocessingService(spark)
    preprocessed_output_df = preprocess_service.process(mock_dcm_output_df)

    if preprocessed_output_df:
        print(f"Preprocessing Service processed {preprocessed_output_df.count()} items.")
        preprocessed_output_df.printSchema()
        preprocessed_output_df.select("gcs_path", "metadata.patient_id", "tensor_shape", "preprocessing_error").show(truncate=False)
    else:
        print("Preprocessing Service did not produce any output.")

    spark.stop()

# --- To test individual services (uncomment the one you want to test): ---
    print("\nTesting Preprocessing Service...")
    main_preprocessing_service_test()
