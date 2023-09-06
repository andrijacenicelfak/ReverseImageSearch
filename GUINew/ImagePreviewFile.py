from functools import reduce
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from GUI.GUIFunctions import *


class ImagePreview(QWidget):
    start_video_player = pyqtSignal(str, list)

    def __init__(
        self,
        description: str,
        image_path: str = None,
        px_image: QPixmap = None,
        width=200,
        height=200,
        text_enabled=True,
        textHeight=50,
        is_video=False,
        frame_num=-1,
        video_path=None,
        classes: str = "",
    ) -> None:
        super().__init__()
        self.setStyleSheet("background-color: #EFEFEF;")
        self.layout_form = QFormLayout()
        self.image_path = image_path
        self.content = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(0)
        self.content_layout.setContentsMargins(0, 2, 2, 0)

        self.image = QLabel(parent=self)
        self.image.setStyleSheet("background-color: #000000;")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMaximumSize(width, height)

        if px_image is not None:
            self.px = px_image
        elif image_path is not None:
            self.px = QPixmap(format_image_path(image_path)).scaled(
                width,
                height,
                aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=Qt.TransformationMode.FastTransformation,
            )

        self.image.setPixmap(self.px)
        self.image.setMinimumSize(int(width), int(height))
        self.image.setMaximumSize(int(width), int(height))
        self.content_layout.addWidget(self.image, stretch=1)
        self.content.setLayout(self.content_layout)

        self.layout_form.addWidget(self.content)
        self.layout_form.setSpacing(0)
        self.layout_form.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout_form)
        self.setToolTip(f"Decription : {description}\nClasses : {classes}")

        if text_enabled:
            self.lbl = QLabel(parent=self, text=description)
            self.lbl.setStyleSheet("")
            self.lbl.setMaximumSize(width, textHeight)
            self.lbl.setWordWrap(True)
            self.content_layout.addWidget(self.lbl, stretch=2)

        self.setMaximumSize(width, height + (textHeight if text_enabled else 0))
        self.setMinimumSize(width, height + (textHeight if text_enabled else 0))

        self.is_video = is_video
        self.video_path = video_path
        self.frame_list = []
        self.add_frame(self.px, frame_num)
        self.setMouseTracking(True)
        self.mouseDoubleClickEvent = self.doubleClicked

    def add_frame(self, img, frame_num):
        if not self.is_video:
            return
        self.frame_list.append((img, frame_num))

    def doubleClicked(self, event):
        if not self.is_video:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.image_path))
        else:
            self.start_video_player.emit(self.video_path, self.frame_list)
    
    def enterEvent(self, a0):
        super().enterEvent(a0)
        self.setStyleSheet("background-color: #9ccfff;")
        self.image.setStyleSheet("background-color: #21476b;")
        color = QColor(0x21, 0x47, 0x6b, 120)
        mod_px = self.px.copy()
        painter = QPainter(mod_px)
        w, h = mod_px.width(), mod_px.height()
        painter.fillRect(0, 0, w, h, color)
        painter.end()
        self.image.setPixmap(mod_px)

    
    def leaveEvent(self, a0) -> None:
        super().leaveEvent(a0)
        self.setStyleSheet("background-color: #EFEFEF;")
        self.image.setStyleSheet("background-color: #000000;")
        self.image.setPixmap(self.px)

