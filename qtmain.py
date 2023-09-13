from functools import reduce
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys

from DB.SqliteDB import ImageDB
from GUINew.AppFile import App
from ImageAnalyzationModule.ConvolutionalModels import *

from ImageAnalyzationModule.ImageAnalyzationFile import *

from ImageAnalyzationModule.Describe import Describe
from ImageAnalyzationModule.Vectorize import Vectorize

from GUI.GUIFunctions import *

HAS_THEME = False

if __name__ == "__main__":
        
    app = QApplication(sys.argv)
    app.setStyleSheet("QLabel{font-size: 14pt;} QLineEdit{font-size: 14pt;} QCheckBox{font-size: 14pt;} QPushButton{font-size: 14pt;} QComboBox{font-size: 14pt;}")
    ai = ImageAnalyzation(
        "yolov8s.pt",
        device="cuda",
        aedType=AutoEncoderDecoderS, coderDecoderModel="3S-NF-29", normalization=False
    )
    db = ImageDB()
    desc=Describe()
    vec=Vectorize()
    window = App(image_analyzation=ai, img_db=db,vec=vec,desc=desc)
    window.show()
    sys.exit(app.exec_())
