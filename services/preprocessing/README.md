# Phase 4: X-Ray Preprocessing Microservice

## Overview
gRPC microservice for preprocessing X-ray images (resize, grayscale, histogram equalization) and storing results in Google Cloud Storage.

## Prerequisites
- Docker
- Python 3
- Google Cloud account
- `gcp-credentials.json` file for google cloud storage authentication

## Deployment

### 1. Build the Docker Image
```bash
docker build -t xray-preprocessor -f Preprocessing.dockerfile .
```

### 2. Run Docker
```bash
docker run -p 5000:5000 \
-v "$(pwd)/preprocessing/gcp-credentials.json:/preprocessing-service/gcp-credentials.json" \
-e GOOGLE_APPLICATION_CREDENTIALS="/preprocessing-service/gcp-credentials.json" \
xray-preprocessor
```

This microservice processes images as they are uploaded



## Setup
1. Install dependencies:
   ```
   pip install -r requirements_preprocessing.txt
   ```
2. Generate python code from preprocessing.proto file:
   ```
   python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. preprocessing.proto
   ```