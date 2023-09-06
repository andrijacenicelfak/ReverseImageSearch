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

try:
    import qdarktheme
    HAS_THEME = True
except ImportError as e:
    pass 

if __name__ == "__main__":
    if HAS_THEME:
        qdarktheme.enable_hi_dpi()
        
    app = QApplication(sys.argv)
    if HAS_THEME:
        qdarktheme.setup_theme("light")

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
