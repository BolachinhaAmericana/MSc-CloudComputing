from reportlab.pdfgen import canvas # For reporting
from reportlab.lib.pagesizes import letter # For reporting
from google.cloud import storage
import numpy as np
from PIL import Image
import os
import io

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import (
    StructType, StructField, StringType, MapType, BinaryType,
    ArrayType, IntegerType, FloatType, Row
)
GCS_REPORT_BUCKET_NAME = "reports-cc-25" 

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
    #depois eliminar esta função daqui pois já existe no preprocessing.py aqui é so pra testar individualmente
    
# --- 4. Reporting Service (based on report.py) ---
def _generate_report_worker(
    original_dicom_gcs_path, metadata_map, 
    image_bytes_for_report, image_shape_list_for_report, image_dtype_str_for_report, 
    prediction_label, confidence_val, 
    report_target_bucket_name=GCS_REPORT_BUCKET_NAME, target_project_id):
    try:
        if image_bytes_for_report is None or not image_shape_list_for_report or metadata_map is None:
            return (None, "Missing data (image, shape, or metadata) for report generation")

        # 1. Reconstruct the original image for the report (can be different from processed tensor)
        img_shape_tuple = tuple(image_shape_list_for_report)
        img_numpy_dtype = _numpy_dtype_from_string(image_dtype_str_for_report)
        pixel_array_for_report = np.frombuffer(image_bytes_for_report, dtype=img_numpy_dtype).reshape(img_shape_tuple)
        
        # Handle multi-frame/channel images - take the first frame/channel for the report display
        if pixel_array_for_report.ndim == 3:
            pixel_array_for_report = pixel_array_for_report[0] 
        if pixel_array_for_report.ndim != 2:
             return (None, f"Report image not 2D after initial processing: shape {pixel_array_for_report.shape}")


        # Convert pixel array to an 8-bit grayscale PIL Image for reportlab
        # (Logic adapted from original report.py's _save_array_as_image)
        img_disp_array = pixel_array_for_report.copy()
        if img_disp_array.dtype != np.uint8: # Needs conversion to 8-bit for 'L' mode PIL
            min_v, max_v = img_disp_array.min(), img_disp_array.max()
            if max_v == min_v:
                img_disp_array = np.zeros_like(img_disp_array, dtype=np.uint8)
            else:
                img_disp_array = ((img_disp_array - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        
        pil_report_image = Image.fromarray(img_disp_array, mode='L')
        
        img_temp_buffer = io.BytesIO()
        pil_report_image.save(img_temp_buffer, format="PNG") # Save as PNG into buffer
        img_temp_buffer.seek(0)

        # 2. Prepare metadata, including inference results
        report_data = dict(metadata_map) # Create a mutable copy
        report_data["Inference_Prediction"] = str(prediction_label) if prediction_label else "N/A"
        report_data["Inference_Confidence"] = f"{confidence_val:.2f}%" if confidence_val is not None else "N/A"
        
        # Determine a unique report filename
        patient_id_for_filename = report_data.get("patient_id", "UnknownPatient")
        base_dicom_filename = os.path.splitext(os.path.basename(original_dicom_gcs_path))[0]
        report_filename_on_gcs = f"reports/report_{patient_id_for_filename}_{base_dicom_filename}.pdf"

        # 3. Generate PDF into a BytesIO buffer
        pdf_buffer = io.BytesIO()
        pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=letter)
        page_width, page_height = letter

        # Add content to PDF (customize as needed)
        pdf_canvas.setFont("Helvetica-Bold", 16)
        pdf_canvas.drawString(72, page_height - 72, "X-Ray Analysis Report")
        
        text_y_start = page_height - 108
        pdf_canvas.setFont("Helvetica", 10)
        
        # Display key information first
        key_info_order = ["patient_id", "patient_name", "study_description", "Inference_Prediction", "Inference_Confidence"]
        for key in key_info_order:
            value = report_data.get(key, "N/A")
            pdf_canvas.drawString(72, text_y_start, f"{key.replace('_', ' ').title()}: {value}")
            text_y_start -= 18
        
        # Add image to PDF (adjust size and position as needed)
        img_display_width, img_display_height = 400, 300 
        img_x_pos = (page_width - img_display_width) / 2
        img_y_pos = text_y_start - img_display_height - 36 # Position below text
        
        if img_y_pos < 72 : # Ensure image is not off page
            img_y_pos = 72
            img_display_height = text_y_start - 108 # shrink height if needed

        try:
            pdf_canvas.drawImage(img_temp_buffer, img_x_pos, img_y_pos, 
                                 width=img_display_width, height=img_display_height, 
                                 preserveAspectRatio=True, anchor='n')
        except Exception as img_err:
            pdf_canvas.setFillColorRGB(1,0,0) # Red text for error
            pdf_canvas.drawString(img_x_pos, img_y_pos + img_display_height/2, f"Error displaying image: {img_err}")
            pdf_canvas.setFillColorRGB(0,0,0) # Back to black

        pdf_canvas.save()
        pdf_buffer.seek(0)

        # 4. Upload PDF to GCS
        gcs_client_report = storage.Client(project=target_project_id)
        bucket_report = gcs_client_report.bucket(report_target_bucket_name)
        blob_report = bucket_report.blob(report_filename_on_gcs)
        blob_report.upload_from_file(pdf_buffer, content_type='application/pdf')
        
        uploaded_report_gcs_url = f"gs://{report_target_bucket_name}/{report_filename_on_gcs}"
        return (uploaded_report_gcs_url, "Report generated and uploaded successfully.")
    except Exception as e:
        # import traceback
        # tb_str = traceback.format_exc()
        return (None, f"Report generation/upload error: {str(e)}")


class SparkReportingService:
    def __init__(self, spark_session, gcs_report_bucket_param, gcs_project_id_param):
        self.spark = spark_session
        self.gcs_report_bucket = gcs_report_bucket_param
        self.gcs_project_id = gcs_project_id_param

    def process(self, input_df_for_reporting):
        if input_df_for_reporting is None:
            print("ReportingService: Input DataFrame is None.")
            return None

        schema = StructType([
            StructField("report_gcs_path", StringType(), True),
            StructField("status_message", StringType(), True)
        ])
        
        report_udf = udf(lambda gcs_path, meta, img_bytes, img_shape, img_dtype, pred, conf: 
            _generate_report_worker(
                gcs_path, meta, img_bytes, img_shape, img_dtype, pred, conf,
                self.gcs_report_bucket, self.gcs_project_id
            ), schema
        )

        # We need original image data for the report, not the tensor
        # Ensure image_data_bytes, image_shape, image_dtype_str from DCM processing are carried through
        reported_df = input_df_for_reporting.withColumn("reporting_result", report_udf(
            col("gcs_path"),       # Original DICOM GCS path
            col("metadata"), 
            col("image_data_bytes"), # Original image bytes from DCM
            col("image_shape"),      # Original image shape from DCM
            col("image_dtype_str"),  # Original image dtype from DCM
            col("prediction"),
            col("confidence")
        ))
        
        final_df = reported_df.select(
            "gcs_path", "metadata.patient_id", "prediction", "confidence", # Key fields for summary
            col("reporting_result.report_gcs_path").alias("report_gcs_path"),
            col("reporting_result.status_message").alias("reporting_status"),
            col("preprocessing_error"), # Carry over any previous errors
            col("inference_error")      # Carry over any previous errors
        )
        return final_df

#pode ter erros, fazer debugging

def main_reporting_service_test():
    spark = SparkSession.builder.appName("ReportingServiceTest").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("--- Testing SparkReportingService ---")
    print(f"Reports will be attempted to be saved to: gs://{GCS_REPORT_BUCKET_NAME}/reports/ in project {GCS_PROJECT_ID}")

    mock_report_image_array = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    mock_report_image_bytes = mock_report_image_array.tobytes()
    mock_report_image_shape = list(mock_report_image_array.shape)
    mock_report_image_dtype = 'uint8'

    mock_inference_data = [
        Row(
            gcs_path="gs://mock-bucket/final_report_test1.dcm",
            metadata={"patient_id": "ReportTest001", "patient_name": "Reporter One", "study_description": "Chest X-Ray", "error": None},
            image_data_bytes=mock_report_image_bytes, 
            image_shape=mock_report_image_shape,     
            image_dtype_str=mock_report_image_dtype, 
            tensor_bytes=b"dummy_tensor_bytes1",      
            tensor_shape=[1,224,224],               
            preprocessing_error=None,
            prediction="PNEUMONIA",
            confidence=95.5,
            inference_error=None
        )
    ]
    mock_inference_schema = StructType([
        StructField("gcs_path", StringType(), True),
        StructField("metadata", MapType(StringType(), StringType()), True),
        StructField("image_data_bytes", BinaryType(), True), 
        StructField("image_shape", ArrayType(IntegerType()), True), 
        StructField("image_dtype_str", StringType(), True), 
        StructField("tensor_bytes", BinaryType(), True), 
        StructField("tensor_shape", ArrayType(IntegerType()), True), 
        StructField("preprocessing_error", StringType(), True),
        StructField("prediction", StringType(), True),
        StructField("confidence", FloatType(), True),
        StructField("inference_error", StringType(), True)
    ])

    mock_inference_output_df = spark.createDataFrame(mock_inference_data, schema=mock_inference_schema)
    print("Mock input to ReportingService:")
    mock_inference_output_df.select("gcs_path", "metadata.patient_id", "prediction", "confidence").show()

    reporting_service = SparkReportingService(spark, GCS_REPORT_BUCKET_NAME, GCS_PROJECT_ID)
    reported_output_df = reporting_service.process(mock_inference_output_df)

    if reported_output_df:
        print(f"Reporting Service processed {reported_output_df.count()} items.")
        reported_output_df.printSchema()
        reported_output_df.show(truncate=False)
    else:
        print("Reporting Service did not produce any output.")

    spark.stop()

print("\nTesting Reporting Service...")
main_reporting_service_test() # Make sure GCS_REPORT_BUCKET_NAME is writable & project ID is correct
