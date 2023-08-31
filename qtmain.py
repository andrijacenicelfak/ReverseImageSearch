from functools import reduce
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys

from DB.SqliteDB import ImageDB
from GUINew.AppFile import App
from ImageAnalyzationModule.ConvolutionalModels import *

from ImageAnalyzationModule.ImageAnalyzationFile import *
from GUI.GUIFunctions import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ai = ImageAnalyzation(
        "yolov8s.pt",
        device="cuda",
        aedType=AutoEncoderDecoderS, coderDecoderModel="3S-NF-29", normalization=False
    )
    db = ImageDB()
    window = App(image_analyzation=ai, img_db=db)
    window.show()
    sys.exit(app.exec_())
