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

def numpy_to_pixmap(numpy_image):
    height, width, channel = numpy_image.shape
    bytes_per_line = 3 * width
    q_image = QImage(numpy_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_image)
    return pixmap

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
        #TODO doubleclick na image
        print(self.image_path)

class ImageGrid(QScrollArea):
    def __init__(self, item_size = 350):
        super().__init__()
        self.content = QWidget()
        self.layout_gird = QGridLayout()
        self.layout_gird.setSpacing(0)
        self.layout_gird.setContentsMargins(0,0,0,0)
        self.item_size = item_size
        self.max_collum_count = max(self.content.width() // self.item_size, 1)
        #TODO: remove this, just example
        # for i in range(16):
        #     ip = ImagePreview(image_path="C:\\Users\\best_intern\\Desktop\\New folder (2)\\1.jpeg", description=f"This is\n the\n {i}th img")
        #     self.layout_gird.addWidget(ip, i // self.max_collum_count, i % self.max_collum_count)

        self.content.setLayout(self.layout_gird)
        self.setWidget(self.content)
        self.setWidgetResizable(True)
        self.resizeEvent = self.on_resize

    def on_resize(self, event):
        old_collum_count = self.max_collum_count
        print(self.content.width())
        self.max_collum_count = max(self.content.width() // self.item_size, 1)

        if self.max_collum_count == old_collum_count:
            return
        
        # widgets = list(enumerate(self.layout_gird))
        widgets = [(i, self.layout_gird.itemAt(i).widget()) for i in range(self.layout_gird.count())]
                       
        while self.layout_gird.count() > 0:
            item = self.layout_gird.itemAt(0)
            if item.widget():
                self.layout_gird.removeWidget(item.widget())
            else:
                self.layout_gird.removeItem(item)
            
        for i, widget in widgets:
            self.layout_gird.addWidget(widget, i // self.max_collum_count, i % self.max_collum_count)
        return

    def removeAllImages(self):
        while self.layout_gird.count():
            item = self.layout_gird.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()


    def addImages(self, data : list[ImageData]):
        self.removeAllImages()
        for i, d in enumerate(data):
            classes = reduce(lambda a, b: f"{a} {b.className}" , d.classes, initial="")
            ip = ImagePreview(d.orgImage, description=f"Path:{d.orgImage}\nClasses: {classes}")
            self.layout_gird.addWidget(ip, i // self.max_collum_count, i % self.max_collum_count)

class SearchImageView(QWidget):
    def __init__(self, width = 400, height = 400) -> None:
        super().__init__()
        self.image_width = width
        self.image_height = height
        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(0,0,0,0)
        self.content = QWidget()
        self.content.setLayout(self.content_layout)
        self.image = None
        self.content_desc = None

        self.layout_form = QFormLayout()
        self.setLayout(self.layout_form)
        self.layout_form.addWidget(self.content)

    def showImage(self, imagePath : str, img_data : ImageData = None):
        #TODO: PASS THROUGH ImageData and load the image before
        # Image ------------------------------------------
        if self.image:
            self.content_layout.removeWidget(self.image)
            self.image.deleteLater()
        self.image = QLabel(parent=self)
        self.image.setMaximumSize(self.image_width, self.image_height)
        # self.px = QPixmap(imagePath).scaled(width, height)
        img = img_data.orgImage
        self.org_img = numpy_to_pixmap(img).scaled(self.image_width, self.image_height)
        self.bb_img = numpy_to_pixmap(drawClasses(self.img_data, img.copy(), fontSize=img.shape[0]//600)).scaled(self.image_width, self.image_height)
        self.image.setPixmap(self.org_img)
        self.content_layout.addWidget(self.image)

        # Content desc ------------------------------------
        if self.content_desc:
            self.content_layout.removeWidget(self.content_desc)
            self.content_desc.deleteLater()
        self.content_desc = QWidget()
        self.content_desc_layout = QVBoxLayout()
        self.content_desc.setLayout(self.content_desc_layout)

        self.path_lbl = QLabel(text=f"Image path : {imagePath}")
        self.path_lbl.setWordWrap(True)
        self.content_desc_layout.addWidget(self.path_lbl)

        self.objects_lbl = QLabel(text=f"Objects : cat, cat, cat, cat")
        self.objects_lbl.setWordWrap(True)
        self.content_desc_layout.addWidget(self.objects_lbl)

        self.draw_bb = QPushButton("Toggle bounding boxes", self)
        self.draw_bb.setCheckable(True)
        self.draw_bb.setChecked(False)
        self.draw_bb.clicked.connect(self.toggleBoundingBox)
        self.content_desc_layout.addWidget(self.draw_bb)
        self.content_layout.addWidget(self.content_desc)


    def toggleBoundingBox(self):
        self.image.setPixmap(self.bb_img if self.draw_bb.isChecked() else self.org_img)
        return

class App(QMainWindow):
    def __init__(self, image_analyzation : ImageAnalyzation):
        super().__init__()
        self.setWindowTitle("App")
        self.content = QWidget()
        self.image_analyzation = image_analyzation
        self.setGeometry(200, 200, 1200, 1200)
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        # self.search_image.showImage(imagePath="C:\\Users\\best_intern\\Desktop\\New folder (2)\\1.jpeg")
        # self.search_image.hide()
        # self.image_grid.hide()
        # Menu bar ----------------------------------------------------------------------------
        self.menu_bar = self.menuBar()

        # File
        self.file_action = QMenu("File", self)

        self.file_search = QAction("Search image", self.file_action)
        self.file_search.triggered.connect(self.file_search_action)
        self.file_action.addAction(self.file_search)

        self.file_add = QAction("Add image", self.file_action)
        self.file_add.triggered.connect(self.file_add_action)
        self.file_action.addAction(self.file_add)

        self.file_add_folder = QAction("Add folder", self.file_action)
        self.file_add_folder.triggered.connect(self.file_add_folder_action)
        self.file_action.addAction(self.file_add_folder)

        self.file_exit = QAction("Exit", self.file_action)
        self.file_exit.triggered.connect(self.file_exit_action)
        self.file_action.addAction(self.file_exit)

        self.menu_bar.addMenu(self.file_action)
        self.menu_bar.addSeparator()
        # Search bar
        self.search_layout = QHBoxLayout()
        self.search_layout.setContentsMargins(0,0,0,0)

        self.search_box = QLineEdit("", self.menu_bar)
        self.search_layout.addWidget(self.search_box)

        self.search_keyword_button = QPushButton("🔍", self.menu_bar)
        self.search_keyword_button.clicked.connect(self.search_keyword_action)
        self.search_layout.addWidget(self.search_keyword_button)

        self.search_image_button = QPushButton("⮹", self.menu_bar)
        self.search_image_button.clicked.connect(self.search_image_action)
        self.search_layout.addWidget(self.search_image_button)

        self.search_widget = QWidget()
        self.search_widget.setLayout(self.search_layout)

        self.main_layout.addWidget(self.search_widget)

        #Main view
        self.search_image = SearchImageView()
        self.main_layout.addWidget(self.search_image)
        self.image_grid = ImageGrid()
        self.main_layout.addWidget(self.image_grid)


        self.content.setLayout(self.main_layout)
        self.setCentralWidget(self.content)

    def file_search_action(self):
        #TODO
        print("search")
        
    def file_add_action(self):
        #TODO
        print("add")

    def file_add_folder_action(self):
        #TODO
        print("folder")
    
    def search_keyword_action(self):
        #TODO
        print(self.search_box.text())

    def search_image_action(self):
        #TODO
        print("Search image")

    def file_exit_action(self):
        QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # ai = ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="1A-27")
    window = App(image_analyzation=None)
    window.show()
    sys.exit(app.exec_())