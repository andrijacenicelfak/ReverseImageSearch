from functools import reduce
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QImage, QDesktopServices
from PyQt5.QtCore import Qt, QUrl
from GUINew.VideoPlayerFile import VideoPlayer

from ImageAnalyzationModule.ImageAnalyzationFile import *
from GUI.GUIFunctions import *


class ImagePreview(QWidget):
    def __init__(
        self,
        description: str,
        image_path: str = None,
        px_image: QPixmap = None,
        width=200,
        height=200,
        text_enabled=True,
        textWidth=150,
    ) -> None:
        super().__init__()
        self.layout_form = QFormLayout()
        self.image_path = image_path
        self.content = QWidget()
        self.content_layout = QGridLayout()
        self.content_layout.setSpacing(0)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.frame_list = list()

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
        self.mouseDoubleClickEvent = self.doubleClicked
    
    def doubleClicked(self, event):
        # TODO : ADD FOR VIDEO
        path, frame_num = format_if_video_path(self.image_path)
        if not path.endswith(".mp4"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            self.start_player(path, (int(frame_num) // 30) * 1000)
    def start_player(self, video_path, position):


        self.video_player = VideoPlayer(fileName=video_path)
        self.video_player.setWindowTitle("Player")
        self.video_player.resize(600, 400)
        self.video_player.mediaPlayer.setPosition(position)
        self.video_player.show() 