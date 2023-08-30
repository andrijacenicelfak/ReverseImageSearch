from functools import reduce
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QImage, QDesktopServices
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from GUINew.VideoPlayerFile import VideoPlayer

from ImageAnalyzationModule.ImageAnalyzationFile import *
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
        textWidth=150,
        is_video = False,
        frame_num = -1,
        video_path = None
    ) -> None:
        super().__init__()
        self.layout_form = QFormLayout()
        self.image_path = image_path
        self.content = QWidget()
        self.content_layout = QGridLayout()
        self.content_layout.setSpacing(0)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        if text_enabled:
            self.lbl = QLabel(parent=self, text=description)
            self.lbl.setMaximumSize(textWidth, height)
            self.lbl.setWordWrap(True)
            self.content_layout.addWidget(self.lbl, 0, 1)

        self.image = QLabel(parent=self)
        self.image.setMaximumSize(width, height)

        if px_image is not None:
            self.px = px_image
        elif image_path is not None:
            self.px = QPixmap(format_image_path(image_path)).scaled(width, height)

        self.image.setPixmap(self.px)
        self.content_layout.addWidget(self.image, 0, 0)
        self.content.setLayout(self.content_layout)

        self.layout_form.addWidget(self.content)
        self.layout_form.setSpacing(0)
        self.layout_form.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout_form)
        self.setToolTip(description)

        self.setMaximumSize(width + textWidth if text_enabled else 0, height)
        self.setMinimumSize(width + textWidth if text_enabled else 0, height)

        self.is_video = is_video
        self.video_path = video_path
        self.frame_list = []
        self.add_frame(self.px, frame_num)

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