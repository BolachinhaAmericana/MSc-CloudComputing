import os
import tempfile
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from google.cloud import storage
from google.oauth2 import service_account
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import logging


CREDENTIALS_PATH = "./secrets.json"
logging.basicConfig(filename='report_log.txt', level=logging.INFO)

def create_authenticated_storage_client():
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return storage.Client(credentials=credentials)


class SparkReportService:
    def __init__(self,spark, app_name="Report Service with Spark", reports_bucket_name="reports-cc-25" ):

        # self.spark = (SparkSession.builder
        #              .appName(app_name)
        #              .config("spark.executor.memory", "4g")
        #              .config("spark.driver.memory", "4g") 
        #              .config("spark.python.worker.memory", "1g")
        #              .config("spark.executor.cores", "2")
        #              .getOrCreate())
        self.spark = spark
        self.reports_bucket_name = reports_bucket_name

        self.spark.sparkContext.setLogLevel("ERROR")

    """ Original Spark implementation with out of memory issues """
    def run_report(self, inference_df):
        if inference_df is None:
            print("No inference dataframe provided")
            return

        
        for col_name in ("image_data_bytes","image_shape","metadata","prediction","confidence","gcs_path"):
            if col_name not in inference_df.columns:
                print(f"[Report ERROR] Missing column: {col_name}")
                return

        
        report_df = inference_df.select(
            "gcs_path",
            "metadata",
            "image_data_bytes",
            "image_shape",
            "prediction",
            "confidence"
        ).coalesce(8)

        total_count = report_df.count()
        # print(f"[Report] Total rows to process: {total} across 8 partitions")

        bucket_name = self.reports_bucket_name

        def process_partition(rows):
            import numpy as np
            from PIL import Image
            from reportlab.lib.units import inch
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            client = create_authenticated_storage_client()
            bucket = client.bucket(bucket_name)
            count = 0

            for row in rows:
                path = row.gcs_path
                meta = row.metadata or {}
                pred = row.prediction
                conf = row.confidence
                img_bytes = row.image_data_bytes
                shape = row.image_shape

                if img_bytes is None or shape is None:
                    print(f"[Report WARNING] Skipping {path} due to missing image_array_bytes/shape")
                    continue
                # Initialize temporary path variables
                tmp_img_path = None
                tmp_pdf_path = None

                try:
                    # Reconstruct image array from bytes and shape
                    arr = np.frombuffer(img_bytes, dtype=np.uint16).reshape(shape)
                    # Normalize to 0-255 for display
                    disp = (arr / arr.max() * 255).astype(np.uint8)
                    del arr  # Free memory immediately after use

                    # Save image as a temporary PNG
                    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    Image.fromarray(disp).save(tmp_img.name)
                    tmp_img.close()
                    tmp_img_path = tmp_img.name
                    del disp  # Free memory immediately

                    # Generate PDF report
                    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    c = canvas.Canvas(tmp_pdf.name, pagesize=letter)
                    w, h = letter

                    c.setFont("Helvetica-Bold", 16)
                    c.drawCentredString(w / 2, h - 1 * inch, "DICOM Report")

                    # Draw metadata
                    c.setFont("Helvetica", 10)
                    y = h - 1.5 * inch
                    info = {
                        "Patient ID": meta.get("patient_id", "N/A"),
                        "Patient Name": meta.get("patient_name", "N/A"),
                        "Study Desc": meta.get("StudyDescription", meta.get("obs", "N/A")),
                        "File": os.path.basename(path)
                    }
                    for k, v in info.items():
                        txt = f"{k}: {v}"
                        if len(txt) > 80:
                            txt = txt[:77] + "..."
                        c.drawString(1 * inch, y, txt)
                        y -= 0.25 * inch
                        if y < h / 2 + 1 * inch:
                            break

                    # Inference results
                    ry = y - 0.5 * inch
                    if ry < 0.5 * inch:
                        ry = 0.5 * inch
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(1 * inch, ry, "Inference Results:")
                    ry -= 0.25 * inch
                    c.setFont("Helvetica", 10)
                    c.drawString(1.2 * inch, ry, f"Prediction: {pred}")
                    ry -= 0.25 * inch
                    c.drawString(1.2 * inch, ry, f"Confidence: {conf:.1f}%")

                    # Add the temporary image to the PDF (optional, if needed)
                    c.drawImage(tmp_img_path, 1 * inch, 1 * inch, width=2 * inch, height=2 * inch)
                    c.save()
                    tmp_pdf_path = tmp_pdf.name

                    # Upload PDF to GCS
                    name = path.replace("gs://", "").replace("/", "_") + ".pdf"
                    blob = bucket.blob(f"reports/{name}")
                    blob.upload_from_filename(tmp_pdf_path)

                    count += 1
                except Exception as e:
                    print(f"[Report ERROR] {path}: {e}")
                finally:
                    # Clean up temporary files
                    for fn in (tmp_img_path, tmp_pdf_path):
                        try:
                            os.unlink(fn)
                        except:
                            pass

            # print(f"[Report] Partition done: {count} reports")
            return [count]

        # Execute and collect results
        per_part = report_df.rdd.mapPartitions(process_partition).collect()
        uploaded = sum(per_part)
        # print(f"[Report] ✅ Uploaded {uploaded}/{total} reports")
        # print(f"[Report] Check gs://{bucket_name}/reports/")
        logging.info(f"Processed {total_count} reports")
    """ Outras tentativas a meter o spark a funcionar (usando json,txt,etc) """

    # def run_report(self, inference_df):
    #     client = create_authenticated_storage_client()
    #     bucket = client.bucket(self.reports_bucket_name)

    #     if inference_df is None:
    #         print("No inference dataframe provided")
    #         return []

    #     rows = (
    #         inference_df.select("metadata", "prediction", "confidence", "gcs_path")
    #         .collect()
    #     )

    #     uploaded_paths = []

    #     for idx, row in enumerate(rows):
    #         metadata = row["metadata"] or {}
    #         prediction = row["prediction"]
    #         confidence = row["confidence"]
    #         gcs_path = row["gcs_path"]

    #         patient_id = metadata.get("patient_id", f"unknown_{idx}")
    #         data_dict = {
    #             "metadata": metadata,
    #             "prediction": prediction,
    #             "confidence": confidence,
    #             "gcs_path": gcs_path,
    #         }

    #         json_bytes = json.dumps(data_dict, indent=2).encode("utf-8")
    #         blob_name = f"json_reports/reports1_{patient_id}.json"
    #         blob = bucket.blob(blob_name)
    #         blob.upload_from_file(io.BytesIO(json_bytes), content_type="application/json")

    #         full_gcs_path = f"gs://{self.reports_bucket_name}/{blob_name}"
    #         uploaded_paths.append(full_gcs_path)

    #         print(f"✅ JSON report uploaded for patient {patient_id}: {full_gcs_path}")

    #     print(f"📄 Total JSON reports uploaded: {len(uploaded_paths)}")
    #     return uploaded_paths

    def stop_spark(self):
        """Stop the Spark session"""
        if self.spark:
            self.spark.stop()
            print("[Report] Spark session stopped.")



def main():
    """Main function to demonstrate the complete pipeline"""
    # It's highly recommended to create ONE SparkSession here and pass it to services.
    # For now, we'll follow the user's structure where each service creates its own.
    
    # --- Configuration (should ideally be centralized or passed) ---
    gcs_dcm_bucket_name = "msc-g21-dcm_data" # From user's main
    gcs_dcm_prefix = ""
    # Model path and output bucket for preprocessing are usually part of those services' configs
    # For inference_service in user's main, it uses default model path in its XrayInferenceHandler
    # For reports_bucket, SparkReportService uses "reports-cc-25" by default.
    os.makedirs("/app/logs", exist_ok=True)
    logging.basicConfig(filename='/app/logs/pipeline_log.txt', level=logging.INFO)

    print("--- Starting Full X-Ray Analysis Pipeline ---")

    
    try:
        from dcm import SparkDicomProcessor
        from preprocessing import SparkImagePreprocessor 
        from inference import SparkInferenceService     
    except ImportError as e:
        print(f"ERROR: Could not import service modules (dcm.py, preprocessing.py, inference.py): {e}")
        print("Please ensure these Python files are in the same directory or in your PYTHONPATH.")
        return

    dicom_processor = None
    preprocessor = None
    inference_service = None
    report_service = None

    app_name = "Full X-Ray Analysis Pipeline"

    spark = (SparkSession.builder
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

    try:
        
        # Step 1: DICOM processing
        print("\n[Pipeline] Step 1: Processing DICOM files...")
        dcm_handler = SparkDicomProcessor(
        spark,
        gcs_bucket_name="msc-g21-dcm_data",
        gcs_prefix="",
        credentials_path="./secrets.json",
        batch_size=10, # 100, 80,50, did not work. 20 worked for 5 batches, I will use 10 for good measure
        max_images=100  # Over 100 images seems to be the limit, try it out tho
    )   

        preprocessor = SparkImagePreprocessor(spark)
        inference_service = SparkInferenceService(spark)
        report_service = SparkReportService(spark)

        # List all DICOM files up to max_images
        all_dicom_files = dcm_handler.list_dicom_files()

        # Split into batches
        batches = [all_dicom_files[i:i + dcm_handler.batch_size] for i in range(0, len(all_dicom_files), dcm_handler.batch_size)]
        
        for batch_num, batch_paths in enumerate(batches, 1):
            print(f"Processing batch {batch_num}/{len(batches)} with {len(batch_paths)} images")
            dcm_df = dcm_handler.process_file_paths(batch_paths)
            preprocessed_df = preprocessor.preprocess_images(dcm_df)
            inference_df = inference_service.run_inference(preprocessed_df)
            report_service.run_report(inference_df)


    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        raise
    finally:
        spark.stop()
        
        

    print('\n[Pipeline] --- X-Ray Analysis Pipeline Finished ---')

if __name__ == "__main__":
    main()
    
