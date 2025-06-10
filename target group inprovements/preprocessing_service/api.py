from flask import Flask, request
from flask_restx import Api, Resource, fields
from preprocess import SparkImagePreprocessor  # Import your preprocessor class
from datetime import datetime
from pyspark.sql import SparkSession
import logging
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Preprocessing Service API', description='API for DICOM Image Preprocessing')

ns = api.namespace('preprocess', description='Preprocessing operations')

process_model = api.model('PreprocessRequest', {
    # You can add more fields if you want to allow custom input
})

@ns.route('/run')
class PreprocessRun(Resource):
    @ns.expect(process_model)
    def post(self):
        """Trigger preprocessing of DICOM images and return output path"""
        # Set up Spark session (tune as needed)
        

        # Run the main logic from preprocess.py
        try:
            # Import here to avoid circular import if needed
            from preprocess import main as preprocess_main

            # Optionally, you can refactor preprocess.py to expose a function that returns the output path
            output_path = preprocess_main()
            
            return {'output_gcs_path': output_path}
        except Exception as e:
            logging.exception("Preprocessing failed")
            return {'error': str(e)}, 500

@ns.route('/status')
class Status(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'ok'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)