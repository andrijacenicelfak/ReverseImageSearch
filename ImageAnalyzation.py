from ultralytics import YOLO
from ultralytics.nn.modules import head
from torchvision import models
import numpy as np

class ImageAnalyzation:
    def __init__(self, model : str, device : str):
        self.model = YOLO(model)
        self.model.to(device)
        self.modelNames = self.model.names
    
    def getImageData(self, image):
        res = self.model.predict(image, verbose=False)
        return [(self.modelNames[int(c.cls)], np.array(c.xyxy.cpu(), dtype="int").flatten()) for r in res for c in r.boxes]