import multiprocessing as mp
import sys
from GUI import GLUE
from DB.SqliteDB import ImageDB
from ImageAnalyzationModule.ImageAnalyzationFile import ImageAnalyzation, AnalyzationType
from PyQt5.QtWidgets import (
    QApplication
)
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
img_analyzer=ImageAnalyzation.ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=ImageAnalyzation.AnalyzationType.CoderDecoder, coderDecoderModel=r"ImageAnalyzation\ConvModelColor4R5C-28.model")
img_db=ImageDB()
app = QApplication(sys.argv)
window = GLUE.GUI(img_db,img_analyzer)
window.show()
sys.exit(app.exec_())