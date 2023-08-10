import sys
from GLUE import GUI
from SqliteDB import ImageDB
import json
from ImageAnalyzation import AnalyzationType, ImageAnalyzation
from PyQt5.QtWidgets import (
    QApplication
)
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

img_analyzer=ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="C:\\dev\\Demo\\ConvModelColor4R5C-28.model")
img_db=ImageDB()
app = QApplication(sys.argv)
# palette = QPalette()
# palette.setColor(QPalette.Window, QColor(53, 53, 53))
# palette.setColor(QPalette.WindowText, Qt.white)
# palette.setColor(QPalette.Base, QColor(25, 25, 25))
# palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
# palette.setColor(QPalette.ToolTipBase, Qt.black)
# palette.setColor(QPalette.ToolTipText, Qt.white)
# palette.setColor(QPalette.Text, Qt.white)
# palette.setColor(QPalette.Button, QColor(53, 53, 53))
# palette.setColor(QPalette.ButtonText, Qt.white)
# palette.setColor(QPalette.BrightText, Qt.red)
# palette.setColor(QPalette.Link, QColor(42, 130, 218))
# palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
# palette.setColor(QPalette.HighlightedText, Qt.black)
# app.setPalette(palette)
window = GUI(img_db,img_analyzer)
window.show()
sys.exit(app.exec_())