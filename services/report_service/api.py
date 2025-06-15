from flask import Flask, request
from flask_restx import Api, Resource, fields
from report import main as report_main
from report import SparkReportGenerator
from datetime import datetime
from pyspark.sql import SparkSession
import logging
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Reporting Service API', description='API for DICOM Image Reproting')

ns = api.namespace('reporting', description='Report operations')

report_model = api.model('ReportRequest', {
    # You can add more fields if you want to allow custom input
})

@ns.route('/run')
class InferenceRun(Resource):
    @ns.expect(report_model)
    def post(self):
        """Trigger inference of DICOM images and return output path"""
        # Set up Spark session (tune as needed)
        

        # Run the main logic from preprocess.py
        try:
            # Import here to avoid circular import if needed
            

            # Optionally, you can refactor preprocess.py to expose a function that returns the output path
            output_path = report_main()
            
            return {'output_gcs_path': output_path}
        except Exception as e:
            logging.exception("Reporting failed")
            return {'error': str(e)}, 500

@ns.route('/status')
class Status(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'ok'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)