from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

from google.cloud import storage
from google.oauth2 import service_account
# import google.auth # Not strictly needed here if using service_account.Credentials directly

import os
from datetime import datetime
import logging
from io import BytesIO

# PDF Generation Library
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import navy, black, red

# Logging and Credentials
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
CREDENTIALS_PATH = os.path.expanduser(os.path.expandvars(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./secrets.json")))

def create_authenticated_storage_client():
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return storage.Client(credentials=credentials)

# --- Worker Function for PDF Generation and Upload ---
def generate_and_upload_pdf_report_worker(
    patient_id: str,
    original_gcs_path: str,
    prediction: str,
    confidence: float,
    metadata: dict, # Expects a map/dict
    report_bucket_name: str,
    report_prefix: str
) -> str:
    """
    Generates a PDF report for a single inference record and uploads it to GCS.
    Returns the GCS path of the uploaded PDF.
    This function runs on Spark workers.
    """
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(name='ReportTitle', parent=styles['h1'], alignment=TA_CENTER, fontSize=18, spaceAfter=20, textColor=navy))
        styles.add(ParagraphStyle(name='SubTitle', parent=styles['h2'], alignment=TA_LEFT, fontSize=12, spaceBefore=10, spaceAfter=5, textColor=black))
        styles.add(ParagraphStyle(name='NormalLeft', parent=styles['Normal'], alignment=TA_LEFT, spaceBefore=2, spaceAfter=2))
        styles.add(ParagraphStyle(name='BoldLeft', parent=styles['NormalLeft'], fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='PredictionHighlight', parent=styles['NormalLeft'], fontSize=14, textColor=red if prediction == "PNEUMONIA" else black, fontName='Helvetica-Bold'))

        story = []

        story.append(Paragraph("X-Ray Inference Report", styles['ReportTitle']))
        story.append(Paragraph(f"<i>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles['Italic']))
        story.append(Spacer(1, 24))

        story.append(Paragraph("Patient & Image Information", styles['SubTitle']))
        story.append(Paragraph(f"<b>Patient ID:</b> {patient_id or 'N/A'}", styles['NormalLeft']))
        
        # Handle potentially long GCS paths
        if original_gcs_path and len(original_gcs_path) > 70:
            display_gcs_path = original_gcs_path[:35] + "..." + original_gcs_path[-30:]
        else:
            display_gcs_path = original_gcs_path or 'N/A'
        story.append(Paragraph(f"<b>Original Image Path:</b> {display_gcs_path}", styles['NormalLeft']))

        if metadata and isinstance(metadata, dict):
            patient_name = metadata.get("patient_name", "N/A")
            obs = metadata.get("obs", "N/A") # SeriesDescription
            story.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['NormalLeft']))
            story.append(Paragraph(f"<b>Observation/Series:</b> {obs}", styles['NormalLeft']))
        else:
            logger.warning(f"Metadata for patient {patient_id} is not a dict or is None: {metadata}")
            story.append(Paragraph("<b>Patient Name:</b> N/A (metadata issue)", styles['NormalLeft']))
            story.append(Paragraph("<b>Observation/Series:</b> N/A (metadata issue)", styles['NormalLeft']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Inference Results", styles['SubTitle']))
        story.append(Paragraph(f"<b>Prediction:</b> <font color='{red if prediction == 'PNEUMONIA' else 'green'}'>{prediction or 'N/A'}</font>", styles['PredictionHighlight']))
        story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%" if confidence is not None else "N/A", styles['NormalLeft']))
        story.append(Spacer(1, 24))
        
        story.append(Paragraph("<i>Disclaimer: This report is auto-generated and for informational purposes only. Consult a qualified medical professional for diagnosis.</i>", styles['Italic']))

        doc.build(story)
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()

        # Upload to GCS
        # GCS client needs to be initialized here, on the worker.
        worker_credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        storage_client = storage.Client(credentials=worker_credentials)
        
        bucket = storage_client.bucket(report_bucket_name)
        
        safe_patient_id = "".join(c if c.isalnum() else "_" for c in str(patient_id or "unknown_patient"))
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        pdf_blob_name = f"{report_prefix.rstrip('/')}/{safe_patient_id}_report_{timestamp}.pdf"
        
        blob = bucket.blob(pdf_blob_name)
        blob.upload_from_string(pdf_bytes, content_type='application/pdf')
        
        uploaded_pdf_gcs_path = f"gs://{report_bucket_name}/{pdf_blob_name}"
        # logger.info(f"Successfully generated and uploaded report: {uploaded_pdf_gcs_path}") # Can be too verbose
        return uploaded_pdf_gcs_path

    except Exception as e:
        logger.error(f"Error in PDF worker for patient {patient_id}, path {original_gcs_path}: {str(e)}", exc_info=True)
        return f"ERROR_PDF_GENERATION_UPLOAD: {str(e)}"


# --- Spark Report Generator Class ---
class SparkReportGenerator:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.spark.sparkContext.setLogLevel("WARN") # UDFs can be chatty with ERROR level

    def generate_reports_from_df(self, inference_results_df, report_bucket_name, report_prefix):
        if inference_results_df is None or inference_results_df.rdd.isEmpty():
            logger.warning("Input DataFrame for report generation is empty or None.")
            return self.spark.createDataFrame([], StructType([StructField("report_gcs_path", StringType(), True)]))


        # Define UDF
        generate_report_udf = udf(
            lambda pid, path, pred, conf, meta: generate_and_upload_pdf_report_worker(
                pid, path, pred, conf, meta, report_bucket_name, report_prefix
            ),
            StringType()
        )

        logger.info("Applying UDF for PDF report generation and upload...")
        reports_df = inference_results_df.withColumn(
            "report_gcs_path",
            generate_report_udf(
                col("patient_id"),
                col("gcs_path"),
                col("prediction"),
                col("confidence"),
                col("metadata")
            )
        )
        
        reports_df.cache()
        count = reports_df.count() # Action to trigger UDFs
        logger.info(f"Report generation UDF applied to {count} records. PDFs are being uploaded.")
        
        return reports_df


# --- Main Pipeline Function (callable by API) ---
def run_report_generation_pipeline(spark_session,
                                   inference_bucket="inference_output1",
                                   inference_prefix="inference_results",
                                   report_bucket="reports-cc-25",
                                   report_prefix_out="reports1"):
    logger.info("Starting report generation pipeline.")

    storage_client = create_authenticated_storage_client()
    
    
    full_prefix_path = inference_prefix.rstrip('/') + "/"
    blobs = storage_client.list_blobs(inference_bucket, prefix=full_prefix_path, delimiter='/') # Use delimiter to list "folders"
    
    # Get actual subdirectories directly under the prefix
    sub_folders = []
    if blobs.prefixes: # Available if delimiter is used
        for prefix_obj in blobs.prefixes:
            # prefix_obj is like "inference_results/20230101120000/"
            # We want "20230101120000"
            folder_name = prefix_obj.replace(full_prefix_path, "").strip('/')
            if folder_name: # Ensure it's not an empty string
                 sub_folders.append(folder_name)

    if not sub_folders:
        # Fallback if delimiter doesn't work as expected or no direct subfolders
        logger.info(f"No direct subfolders found with delimiter under gs://{inference_bucket}/{full_prefix_path}. Trying to parse all blobs.")
        all_blobs_in_prefix = storage_client.list_blobs(inference_bucket, prefix=full_prefix_path)
        parsed_folders = set()
        for blob in all_blobs_in_prefix:
            # blob.name is like "inference_results/20230101120000/part-00000-....parquet"
            # We want to extract "20230101120000"
            relative_path = blob.name.replace(full_prefix_path, "")
            if '/' in relative_path:
                potential_folder = relative_path.split('/')[0]
                if potential_folder:
                    parsed_folders.add(potential_folder)
        sub_folders = list(parsed_folders)

    if not sub_folders:
        error_msg = f"No inference data sub-folders found in gs://{inference_bucket}/{full_prefix_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    latest_folder_name = sorted(sub_folders)[-1]
    input_parquet_path = f"gs://{inference_bucket}/{full_prefix_path}{latest_folder_name}"
    logger.info(f"Reading inference data from latest folder: {input_parquet_path}")

    inference_df = spark_session.read.parquet(input_parquet_path)

    if inference_df.rdd.isEmpty():
        error_msg = f"No data found in {input_parquet_path}."
        logger.warning(error_msg)
        raise ValueError(error_msg)

    required_cols = ["patient_id", "gcs_path", "prediction", "confidence", "metadata"]
    missing_cols = [r_col for r_col in required_cols if r_col not in inference_df.columns]
    if missing_cols:
        error_msg = f"Input DataFrame from {input_parquet_path} is missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    report_generator = SparkReportGenerator(spark_session=spark_session)
    reports_with_paths_df = report_generator.generate_reports_from_df(
        inference_df, report_bucket, report_prefix_out
    )

    if reports_with_paths_df is None:
        logger.warning("Report generation did not produce a result DataFrame.")
        return []

    # Collect paths of successfully generated PDFs
    # Filter out rows where UDF might have returned an error string
    generated_report_paths_rows = reports_with_paths_df.select("report_gcs_path").collect()
    generated_report_paths = [
        row.report_gcs_path for row in generated_report_paths_rows 
        if row.report_gcs_path and not row.report_gcs_path.startswith("ERROR_")
    ]
    
    successful_reports_count = len(generated_report_paths)
    total_records_processed = reports_with_paths_df.count() # This was already counted
    
    logger.info(f"Successfully generated and initiated upload for {successful_reports_count} PDF reports out of {total_records_processed} records.")
    
    # Log a few example paths
    for path in generated_report_paths[:min(3, len(generated_report_paths))]:
        logger.info(f"Example generated report GCS path: {path}")

    reports_with_paths_df.unpersist()
    return generated_report_paths


# --- Main function for standalone execution ---
def main():
    spark = None
    try:
        spark = (SparkSession.builder
                 .appName("PDF_Report_Generation_Pipeline_Standalone")
                 .config("spark.driver.memory", "4g")
                 .config("spark.executor.memory", "2g")
                 .config("spark.python.worker.memory", "1g")
                 .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                 # Add GCS connector and other necessary Spark configs if running locally against GCS
                 # .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.23.2,com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.0") # Example
                 .getOrCreate())
        logger.info("Spark session created for standalone report generation.")
        
        generated_paths = run_report_generation_pipeline(spark_session=spark)
        
        if generated_paths:
            logger.info(f"Standalone report generation process completed. {len(generated_paths)} reports were processed for upload.")
        else:
            logger.info("No reports were processed for upload or an issue occurred.")

    except Exception as e:
        logger.error(f"An error occurred in the standalone report generation pipeline: {e}", exc_info=True)
    finally:
        if spark:
            logger.info("Stopping Spark session for standalone report generation.")
            spark.stop()
            
    return generated_paths

if __name__ == "__main__":
    # Configure root logger for more detailed output if running standalone
    logging.getLogger().setLevel(logging.INFO) 
    # Add a handler for console output if not already configured by basicConfig
    if not logging.getLogger().handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(console_handler)
    main()