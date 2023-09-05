from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import Qt, qInstallMessageHandler, QObject, QThread, pyqtSignal
from functools import reduce
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import sys

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QCoreApplication
from GUINew.ImagePreviewFile import ImagePreview

from ImageAnalyzationModule.ImageAnalyzationDataTypes import *
from GUI.GUIFunctions import *

IMAGE_SIZE = 200


class ImageGrid(QScrollArea):
    start_video_player = pyqtSignal(str, list)

    def __init__(self, item_size=200, text_enabled=True, loading_percent_callback=None):
        super().__init__()
        self.content = QWidget()
        self.layout_gird = QGridLayout()
        self.layout_gird.setSpacing(0)
        self.layout_gird.setContentsMargins(2, 2, 2, 2)
        self.text_enabled = text_enabled
        self.item_size = item_size + 4
        self.item_height = item_size + (int(item_size * 2 / 3) if text_enabled else 0) + 4
        self.max_collum_count = max(self.content.width() // self.item_size, 1)

        self.content.setLayout(self.layout_gird)
        self.setWidget(self.content)
        self.setWidgetResizable(True)
        self.old_resize = self.resizeEvent
        self.resizeEvent = self.on_resize
        self.loading_percent_callback = loading_percent_callback

    @QtCore.pyqtSlot(str, list)
    def start_video_player_call(self, path:str, data:list):
        self.start_video_player.emit(path, data)

    def on_resize(self, event):
        self.old_resize(event)
        self.content.setMaximumWidth(self.width())
        old_collum_count = self.max_collum_count
        self.max_collum_count = max(self.content.width() // self.item_size, 1)

        if self.max_collum_count == old_collum_count:
            return

        # widgets = list(enumerate(self.layout_gird))
        widgets = [
            (i, self.layout_gird.itemAt(i).widget())
            for i in range(self.layout_gird.count())
        ]

        while self.layout_gird.count() > 0:
            item = self.layout_gird.itemAt(0)
            if item.widget():
                self.layout_gird.removeWidget(item.widget())
            else:
                self.layout_gird.removeItem(item)

        for i, widget in widgets:
            self.layout_gird.addWidget(
                widget, i // self.max_collum_count, i % self.max_collum_count
            )
        return

    def removeAllImages(self):
        while self.layout_gird.count():
            item = self.layout_gird.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def addImages(self, data: list[ImageData]):
        self.removeAllImages()
        data_len = len(data)
        for d in enumerate(data):
            self.add_image(d)
            if self.loading_percent_callback is not None:
                self.loading_percent_callback(100 * d[0] / max(data_len, 1))

    def add_image(self, data):
        i, d = None, None
        if type(data) is tuple:
            i, d = data
        else:
            d = data
            i = self.layout_gird.count()
        classes = reduce((lambda a, b: a + " " + b.className), d.classes, "")
        ip = ImagePreview(
            d.orgImage,
            description=f"Classes: {classes}",
            text_enabled = self.text_enabled,
        )
        self.layout_gird.addWidget(
            ip, i // self.max_collum_count, i % self.max_collum_count
        )

    def add_to_grid(self, data : [tuple]):        
        for image, desc, path, classes in data:
            video = format_if_video_path(path)
            if video[1] is None:
                i = self.layout_gird.count()
                ip = ImagePreview(desc, path, image, classes = classes)
                self.layout_gird.addWidget(
                    ip, i // self.max_collum_count, i % self.max_collum_count
                )
            elif video[0] in self.video_dictonary:
                self.video_dictonary[video[0]].add_frame(image, video[1])
            else:
                i = self.layout_gird.count()
                ip = ImagePreview(desc, path, image, is_video=True, frame_num=video[1], video_path = video[0], classes = classes)
                self.layout_gird.addWidget(
                    ip, i // self.max_collum_count, i % self.max_collum_count
                )
                self.video_dictonary[video[0]] = ip
                ip.start_video_player.connect(self.start_video_player_call)


    def done_adding(self):
        print("Done")

    def add_images_mt(self, data: list[ImageData]):
        self.removeAllImages()
        self.image_adder = ImageAddWorker(data)
        self.image_adder.progress.connect(self.loading_percent_callback)
        self.image_adder.add.connect(self.add_to_grid)
        self.image_adder.done.connect(self.done_adding)

        # video dictonary has the first ImagePreview
        # video_dictonary[key] = original_image_preview
        self.video_dictonary = dict()

        self.image_adder.start()


class ImageAddWorker(QThread):
    done = pyqtSignal()
    progress = pyqtSignal(int)
    add = pyqtSignal(list)

    def __init__(self, data: [ImageData]):
        super().__init__()
        self.data = data

    def run(self):
        data_len = len(self.data)
        data_emit = []
        for i, d in enumerate(self.data):
            classes = reduce((lambda a, b: a + " " + b.className), d.classes, "")
            path = format_image_path(d.orgImage)
            px = QPixmap(path)
            if px.isNull():
                #TODO: Image has been deleted, delete from database
                print("Image deleted : " + path)
                continue
            
            px = px.scaled(IMAGE_SIZE, IMAGE_SIZE, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, transformMode=Qt.TransformationMode.FastTransformation)
            data_emit.append((px, d.description, d.orgImage, classes))
            if i % 5 == 0:
                self.add.emit(data_emit)
                data_emit = []
                self.msleep(1)
                self.progress.emit(int(100 * i / data_len))
        self.add.emit(data_emit)
        self.progress.emit(100)
        self.done.emit()