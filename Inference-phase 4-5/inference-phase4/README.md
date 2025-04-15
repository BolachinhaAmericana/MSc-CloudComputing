
---

### Inference Service Documentation

#### Functional Requirements
- **Receive Input**: Accept an input image (e.g., an X-ray image from a .DICOM file) for processing.
- **Preprocessing**: Process the input image to prepare it for machine learning model inference (e.g., resizing, normalization).
- **Inference**: Execute the machine learning model on the preprocessed image to generate predictions and confidence scores.
- **Output Results**: Provide the inference results, including the prediction and confidence score, to other services (e.g., the report generation service).

#### Microservices Components
- **API**
  - **POST /inference**: Accepts the input image and any optional parameters, triggers preprocessing and inference, and returns the inference results (prediction and confidence score).
- **Preprocessing**
  - Handles the transformation of the input image into a format suitable for the machine learning model (e.g., resizing, normalization, or converting from .DICOM to a usable image format).
- **Inference**
  - Runs the machine learning model on the preprocessed image to produce predictions and associated confidence scores.
- **Storage**
  - (Optional) Temporarily stores the input image or preprocessed image if needed. Note: Primary storage of .DICOM files and metadata is assumed to be handled by another service.

#### Interaction with Other Services
- **Report Generation Service**: Sends the inference results (prediction and confidence score) to the report generation service, where they are combined with patient metadata to create the final report.
- **Input Source**: Receives the .DICOM file or extracted image from an upstream service or user input mechanism.

#### Example Workflow
1. **Input Submission**: The user inputs a .DICOM X-ray file.
2. **Image Extraction and Preprocessing**: The .DICOM file is processed to extract the image, which is then passed to the inference service for preprocessing (e.g., resizing or normalization).
3. **Model Inference**: The preprocessed image is fed into the machine learning model, which generates a prediction and confidence score.
4. **Result Transmission**: The inference results (prediction and confidence score) are sent to the report generation service.
5. **Report Creation**: The report generation service merges the inference results with the previously saved patient metadata (e.g., in JSON format) to generate the final report.


#### Note:
As of now we have our workflow planned and app running purely on python scripts, but we have nor implemented the communications between our services while contained.

---
