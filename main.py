import sys
from GLUE import GUI
from SqliteDB import ImageDB
import json
from ImageAnalyzation import AnalyzationType, ImageAnalyzation
from PyQt5.QtWidgets import (
    QApplication
)

img_analyzer=ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="C:\\dev\\Demo\\ConvModelColor4R5C-28.model")
img_db=ImageDB()
app = QApplication(sys.argv)
window = GUI(img_db,img_analyzer)
window.show()
sys.exit(app.exec_())
