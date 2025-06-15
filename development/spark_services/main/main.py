from inference import *
from report import *
from preprocessing import *
from dcm import *
    

# --- Main Pipeline Orchestration(to be tested) ---

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


if __name__ == "__main__":
    main()