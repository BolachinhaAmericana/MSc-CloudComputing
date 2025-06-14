from flask import Flask, request
from flask_restx import Api, Resource, fields
from dcm import SparkDicomProcessor 
from datetime import datetime  

app = Flask(__name__)
api = Api(app, version='1.0', title='DICOM Service API', description='API for DICOM Processing')

ns = api.namespace('dcm', description='DICOM operations')

process_model = api.model('ProcessRequest', {
    # 'gcs_bucket_name': fields.String(required=True, description='GCS bucket name'),  # Remove from model
    'gcs_prefix': fields.String(required=False, description='GCS prefix', default=''),
})

@ns.route('/process')
class DicomProcess(Resource):
    @ns.expect(process_model)
    def post(self):
        """Trigger DICOM processing and return output path"""
        data = api.payload
        gcs_bucket_name = "msc-g21-dcm_data"  # Hardcoded bucket name for testing(ideally in the deployed app the users can choose from whic data lake they want to acess)
        gcs_prefix = data.get('gcs_prefix', '')

        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder
            .appName("DICOM API")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.executor.cores", "2")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        processor = SparkDicomProcessor(spark=spark, gcs_bucket_name=gcs_bucket_name, gcs_prefix=gcs_prefix)
        final_df = processor.process_dicoms()
        if final_df is None:
            return {'message': 'No DICOM files found.'}, 404

        output_gcs_path = f"gs://dcm_output/processed_dicoms/api_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        final_df.write.mode("overwrite").parquet(output_gcs_path)
        processor.stop_spark()
        return {'output_gcs_path': output_gcs_path}

@ns.route('/status')
class Status(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'ok'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)