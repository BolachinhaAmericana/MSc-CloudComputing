import torch 
import torch.nn.functional as F
import torchxrayvision as xrv
from google.cloud import storage
import google.auth
import os
import tempfile

class InferenceHandler:
    """  Input: Tensor, requested from Preprocessing service and model path (either local or eventually from google cloud storage(i tried locally first thats why its like this))
        Output : List containing the predicted class and the confidence of the model in the prediction
    
    """
    
    def __init__(self,bucket_name="pneumonia-models", model_blob_path="modelFINAL02.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(bucket_name,model_blob_path)
        self.model.eval()
        
    def _load_model(self,bucket_name, model_blob_path):
        """Load the model from the Google Cloud Storage."""
        credentials, project = google.auth.default() # Used in development to bypass authtications
        print(f"Using service account: {credentials.service_account_email}")
        print(f"Attempting to access bucket: '{bucket_name}'")
        print(f"Looking for blob: '{model_blob_path}'")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_blob_path)
        print(f"Bucket exists: {bucket.exists()}")  # Should be True
        print(f"Blob exists: {blob.exists()}")   
        
        with tempfile.NamedTemporaryFile(delete=False,suffix='.pth') as temp_file:
            blob.download_to_file(temp_file)
            temp_file_path = temp_file.name
            
        
        model = xrv.models.DenseNet(weights="densenet121-res224-rsna", op_threshs=None)
        model.eval()
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, 2)
        model.op_threshs = None
        model.to(self.device)
        
        
        # Fix key mismatches(isto é importante )
        new_state_dict = {}
        state_dict = torch.load(temp_file_path, map_location=self.device)
        for k, v in state_dict.items():
            if k.startswith("classifier.1."):  # Handle torchxrayvision format (o xrv apteceu ter um formato diferente do torch)
                new_k = k.replace("classifier.1.", "classifier.")
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
                
        os.remove(temp_file_path)
                
        return model
    
    def predict(self, processed_tensor: torch.Tensor) -> tuple[str,float]:
        """ Predict the class of the input image tensor."""
        processed_tensor = processed_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(processed_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence,predicted_class = torch.max(probabilities, dim=1)

        if predicted_class.item() == 0:
            predicted_class = "NORMAL"
        else:
            predicted_class = "PNEUMONIA"
        
        return predicted_class, (confidence.item()*100) # e.g., ("NORMAL", 95.0)
       
        
            
            
            
            
            