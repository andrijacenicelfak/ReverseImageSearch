from functools import reduce
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from ImageAnalyzationModule.ImageAnalyzationFile import *
from GUI.GUIFunctions import *

class ImagePreview(QWidget):
    def __init__(self, image_path, description, width = 200, height = 200, textWidth = 150) -> None:
        super().__init__()
        self.layout_form = QFormLayout()
        self.image_path = image_path
        self.content = QWidget()
        self.content_layout = QGridLayout()
        self.content_layout.setSpacing(0)
        self.content_layout.setContentsMargins(0,0,0,0)

        self.lbl = QLabel(parent=self, text=description)
        self.lbl.setMaximumSize(textWidth, height)
        self.lbl.setWordWrap(True)

        self.image = QLabel(parent=self)
        self.image.setMaximumSize(width, height)
        self.px = QPixmap(image_path).scaled(width, height)
        self.image.setPixmap(self.px)

        self.content_layout.addWidget(self.image, 0, 0)
        self.content_layout.addWidget(self.lbl, 0, 1)
        self.content.setLayout(self.content_layout)

        self.layout_form.addWidget(self.content)
        self.layout_form.setSpacing(0)
        self.layout_form.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout_form)

        self.setMaximumSize(width + textWidth, height)
        self.setMinimumSize(width + textWidth, height)
        self.mouseDoubleClickEvent = self.doubleClicked
    
    def doubleClicked(self, event):
        print(self.image_path)