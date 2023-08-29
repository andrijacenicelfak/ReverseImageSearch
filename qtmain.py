from functools import reduce
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys

from DB.SqliteDB import ImageDB
from GUINew.AppFile import App
from ImageAnalyzationModule.ConvolutionalModels import AutoEncoderDecoderXS

from ImageAnalyzationModule.ImageAnalyzationFile import *
from GUI.GUIFunctions import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ai = ImageAnalyzation(
        "yolov8s.pt",
        device="cuda",
        analyzationType=AnalyzationType.CoderDecoder,
        aedType=AutoEncoderDecoderXS,
        coderDecoderModel="1XS-15",
    )
    db = ImageDB()
    window = App(image_analyzation=ai, img_db=db)
    window.show()
    sys.exit(app.exec_())
