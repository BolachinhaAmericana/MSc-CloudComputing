from flask import Flask, request
from flask_restx import Api, Resource, fields
from inference import SparkInferenceService, XrayInferenceHandler  # Import your preprocessor class
from datetime import datetime
from pyspark.sql import SparkSession
import logging
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Inference Service API', description='API for DICOM Image Inference')

ns = api.namespace('inference', description='Inference operations')

inference_model = api.model('InferenceRequest', {
    # You can add more fields if you want to allow custom input
})

@ns.route('/run')
class InferenceRun(Resource):
    @ns.expect(inference_model)
    def post(self):
        """Trigger inference of DICOM images and return output path"""
        # Set up Spark session (tune as needed)
        

        # Run the main logic from preprocess.py
        try:
            # Import here to avoid circular import if needed
            from inference import main as inference_main

            # Optionally, you can refactor preprocess.py to expose a function that returns the output path
            output_path = inference_main()
            
            return {'output_gcs_path': output_path}
        except Exception as e:
            logging.exception("Inference failed")
            return {'error': str(e)}, 500

@ns.route('/status')
class Status(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'ok'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)