from functools import reduce
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from GUINew.AppFile import App

from ImageAnalyzationModule.ImageAnalyzationFile import *
from GUI.GUIFunctions import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ai = ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="1A-27")
    window = App(image_analyzation=ai)
    window.show()
    sys.exit(app.exec_())