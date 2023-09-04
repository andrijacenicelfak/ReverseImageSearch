import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
import multiprocessing as mp

class Describe():
    def __init__(self,device="cuda"):
        self.device=device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)       
    def caption(self,image_path):
        img=cv2.cvtColor(image_path,cv2.COLOR_BGR2RGB)
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0],skip_special_tokens=True)