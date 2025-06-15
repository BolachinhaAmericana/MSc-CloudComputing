from flask import Flask, request
from flask_restx import Api, Resource, fields
from dcm import SparkDicomProcessor 
from dcm import main_dcm as dcm_main 
from datetime import datetime  

app = Flask(__name__)
api = Api(app, version='1.0', title='DICOM Service API', description='API for DICOM Processing')

ns = api.namespace('dcm', description='DICOM operations')

process_model = api.model('ProcessRequest', {
    # 'gcs_bucket_name': fields.String(required=True, description='GCS bucket name'),  # Remove from model
    # 'gcs_prefix': fields.String(required=False, description='GCS prefix', default=''),
})

@ns.route('/process')
class DicomProcess(Resource):
    @ns.expect(process_model)
    def post(self):
        """Trigger DICOM processing and return output path"""
        output_gcs_path = None
        try:
            

            

            # Run the main logic from dcm.py
            output_gcs_path = dcm_main()

        except Exception as e:
            app.logger.error(f"DICOM processing failed: {str(e)}")
            return {'error': str(e)}, 500
        return {'output_gcs_path': output_gcs_path}

@ns.route('/status')
class Status(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'ok'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)