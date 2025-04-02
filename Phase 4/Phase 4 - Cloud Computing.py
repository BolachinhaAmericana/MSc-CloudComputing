from torchvision.transforms import v2 as transforms
from PIL import Image
import io
import cv2
import numpy as np
from google.cloud import storage

class Preprocessor:
    def __init__(self, bucket_name="xray-bucket"):
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: Image.fromarray(
                cv2.GaussianBlur(np.array(x), (5, 5), 0)  
            )),
            transforms.Lambda(lambda x: Image.fromarray(
                cv2.equalizeHist(np.array(x)) if len(np.array(x).shape) == 2  
                else cv2.merge([cv2.equalizeHist(ch) for ch in cv2.split(np.array(x))])
            )),
            transforms.Lambda(lambda x: self._crop_lung_region(x)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
         ])
        self.bucket = storage.Client().bucket(bucket_name)

    def process(self, image_bytes, user_id):
        image = Image.open(io.BytesIO(image_bytes))
        #processed_image = self.transform(image)
        tensor = self.transform(image).unsqueeze(0)

        image.save(self.bucket.blob(f"{user_id}.png").open("wb"), format="PNG")
        
        return tensor, image
       