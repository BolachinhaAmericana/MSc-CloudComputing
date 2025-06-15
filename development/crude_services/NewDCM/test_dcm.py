import os
import sys
import io
import pydicom
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BinaryType, ArrayType

from google.cloud import storage
import google
from google.oauth2 import service_account

class Configurations:
    def __init__(self):
        self.project_id = "cloud-computing-project-2025"
        self.bucket_name = "msc-g21-dcm_data"
        self.credentials = service_account.Credentials.from_service_account_file('./secrets.json')


class DicomGCSProcessor(Configurations):
    def __init__(self):
        super().__init__()
        self.storage_client = storage.Client(project=self.project_id)
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("DICOM-GCS-Processor") \
            .config("spark.ui.port", "4041") \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-gcs:3.3.1") \
            .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
            .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", self.credentials) \
            .getOrCreate()

        self.client = storage.Client(project=self.project_id, credentials=self.credentials)
        self.bucket = self.client.bucket(self.bucket_name)

if __name__ == "__main__":
    print("running...")

    print("Python version:", sys.version)
    print("SPARK_HOME:", os.environ.get('SPARK_HOME'))
    print("JAVA_HOME:", os.environ.get('JAVA_HOME'))
    print("PATH:", os.environ.get('PATH'))

    processor = DicomGCSProcessor()

    #