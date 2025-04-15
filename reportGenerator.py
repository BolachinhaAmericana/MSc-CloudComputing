import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from google.cloud import storage
import tempfile
from PIL import Image
import os
import io


#from services.DcmHandler.MegaDCM import UseDicom

class ReportGenerator():
    def __init__(self, image_pixel_array, dicom_metadata, bucket_name="reports-cc-25"):
        self.img = image_pixel_array
        self.metadata = dicom_metadata
        self.bucket_name = bucket_name

    def _save_array_as_image(self):
        """Convert the pixel array to an image file and return the path."""
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        img_array = self.img.copy()
        
        # Handle different bit depths
        if img_array.dtype == np.uint16:
            # Scale 16-bit to 8-bit
            img_array = (img_array / 256).astype(np.uint8)
        elif img_array.max() > 255:
            # Normalize to 0-255 range
            img_array = ((img_array - img_array.min()) * 255 / 
                        (img_array.max() - img_array.min())).astype(np.uint8)
        
        # Make sure array is 8-bit for PIL
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        
        img = Image.fromarray(img_array)
        #Novo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            img.save(temp_file.name, format='PNG')
            return temp_file.name
        
        #Antigo quando gravamos localmente
        # img.save(temp_path, format='PNG')
        # print(f"Temporary image saved to {temp_path}")
        # return temp_path
        
    def generate_report(self):
        temp_image_path = self._save_array_as_image()
        
        # Generate PDF
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(self.out_path, pagesize=letter)
        width, height = letter
        
        # Add metadata
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 50, f"DICOM Report")
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Patient ID: {self.metadata.get('patient_id', '')}")
        
        y_position = height - 90
        c.setFont("Helvetica", 9)
        
        # Draw key metadata items
        for key, value in self.metadata.items():
            if y_position < 400:  # Leave space for the image
                break
            c.drawString(50, y_position, f"{key}: {value}")
            y_position -= 15
        
        # Position image
        img_width = 450 
        img_height = 300
        img_x = (width - img_width) / 2  # Center horizontally
        img_y = 50
        
        # Add image to PDF
        try:
            c.drawImage(temp_image_path, img_x, img_y, width=img_width, height=img_height)
        except Exception as e:
            print(f"Error adding image to PDF: {e}")
            # Add error message to PDF
            c.setFont("Helvetica-Bold", 10)
            c.setFillColorRGB(1, 0, 0)  # Red text
            c.drawString(50, img_y + 150, f"Error displaying image: {str(e)}")
        
        # Finalize PDF
        c.save()
        
        pdf_buffer.seek(0)
        
        # Upload to Google Cloud Storage
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(f"reports/report_{self.metadata.get('patient_id', 'unknown')}.pdf")
        blob.upload_from_file(pdf_buffer, content_type='application/pdf')
        
        # Generate a signed URL (valid for 1 hour)
        url = blob.generate_signed_url(expiration=3600)
        
        
        
        # Clean up temporary file
        os.unlink(temp_image_path)
        
        print(f"Report saved in: {url}")
        
        return url
    
    
if __name__ == "__main__":
    
    """
    dcm_path='./temp.dcm'
    dcm = UseDicom(dcm_path)
    img = dcm.get_image_pixel_array()
    meta = dcm.get_metadata()

    ReportGenerator(img, meta, './test.pdf').generate_report()"
    """

