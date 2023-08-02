import time
from SqliteDB import ImageDB
from ImageAnalyzation import ImageAnalyzation

img_analyzer=ImageAnalyzation("yolov8s.pt","cuda")
img_db=ImageDB()

