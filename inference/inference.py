import torch 
import torch.nn.functional as F
import torchxrayvision as xrv

class InferenceHandler:
    """  """
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str):
        """Load the model from the specified path."""
        # with torch.serialization.safe_globals([xrv.models.DenseNet]):
        model = xrv.models.DenseNet(weights="densenet121-res224-rsna", op_threshs=None)
        model.eval()
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, 2)
        model.op_threshs = None
        model.to(self.device)
        # Fix key mismatches(isto é importante )
        new_state_dict = {}
        state_dict = torch.load(model_path, map_location=self.device)
        for k, v in state_dict.items():
            if k.startswith("classifier.1."):  # Handle torchxrayvision format (o xrv apteceu ter um formato diferente do torch)
                new_k = k.replace("classifier.1.", "classifier.")
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
                
        return model
    
    def predict(self, processed_tensor: torch.Tensor) -> torch.Tensor:
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
        
        return [predicted_class, (confidence.item()*100)] # e.g predicted_class = 0, confidence = 0.95 (95% confidence)
       
        ## Tbm posso fazer o output voltar pro cpu so pra ter a certeza que funciona 
            
            
            
            
            
