import array
from functools import reduce
import pathlib
import typing
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import (
    QObject,
    Qt,
    QThread,
    pyqtSignal,
    pyqtSlot,
    qInstallMessageHandler,
    QtMsgType,
    QMessageLogContext,
)
from PyQt5.QtGui import QIcon
from DB.SqliteDB import ImageDB
import DB.Functions as dbf
from GUINew.VideoPlayerFile import VideoPlayer
from GUINew.DisplayFile import *
from GUINew.ImageGridFile import ImageGrid
from GUINew.SearchImageDialogFile import SearchImageDialog
from GUINew.SearchImageFile import *
from ImageAnalyzationModule.ImageAnalyzationFile import *
from ImageAnalyzationModule.ImageAnalyzationDataTypes import *
from GUI.GUIFunctions import *
import multiprocessing as mp
from FileSystem.FileExplorerFile import search

from Video.histoDThresh import summ_video_parallel
from GUINew.ThreadsFile import *
import os
from GUINew.IndexFunctions import IndexFunction

from ImageAnalyzationModule.Vectorize import Vectorize
from ImageAnalyzationModule.Describe import Describe

VIDEO_THUMBNAIL_SIZE = 200
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi")


def handle(type: QtMsgType, context: QMessageLogContext, message: str):
    if context.category == "qt.gui.icc":
        return
    print(f"type : {type}, category : {context.category}, message : {message}")


class App(QMainWindow):
    def __init__(
        self,
        image_analyzation: ImageAnalyzation,
        img_db: ImageDB,
        desc: Describe,
        vec: Vectorize,
    ):
        super().__init__()
        self.setWindowIcon(QIcon(".\AppImages\search.png"))
        self.setWindowTitle("Kimaris")
        self.video_player = None
        #
        self.index_worker = None
        qInstallMessageHandler(handle)

        #
        self.content = QWidget()
        self.image_analyzation = image_analyzation
        self.img_db = img_db
        self.desc = desc
        self.vec = vec
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Menu bar ----------------------------------------------------------------------------
        self.menu_bar = self.menuBar()

        # File
        self.file_action = QMenu("File", self)

        self.file_search = QAction("Search image", self.file_action)
        self.file_search.triggered.connect(self.search_image_action)
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

        # Settings
        self.settings_action = QAction("Settings", self.menu_bar)
        self.settings_action.triggered.connect(self.settings_change)
        self.menu_bar.addAction(self.settings_action)

        # Search bar ------------------------------------------------------------------------
        self.search_layout = QHBoxLayout()
        self.search_layout.setContentsMargins(0, 0, 0, 0)

        self.search_box = QLineEdit("", self.menu_bar)
        self.search_box.returnPressed.connect(self.enter_search_box)
        self.search_layout.addWidget(self.search_box)

        self.search_keyword_button = QPushButton("", self.menu_bar)
        self.search_keyword_button.setIcon(QIcon(".\\AppImages\\search.png"))
        self.search_keyword_button.clicked.connect(self.search_keyword_action)
        self.search_layout.addWidget(self.search_keyword_button)

        self.search_image_button = QPushButton("", self.menu_bar)
        self.search_image_button.setIcon(QIcon(".\\AppImages\\image.png"))
        self.search_image_button.clicked.connect(self.search_image_action)
        self.search_layout.addWidget(self.search_image_button)

        self.search_widget = QWidget()
        self.search_widget.setLayout(self.search_layout)

        self.main_layout.addWidget(self.search_widget)

        # Main view
        self.search_image = SearchImageView()
        self.main_layout.addWidget(self.search_image)
        self.image_grid = ImageGrid(loading_percent_callback=self.set_loading_percent)
        self.image_grid.start_video_player.connect(self.start_video_player)
        self.main_layout.addWidget(self.image_grid)

        # loader
        self.loading_percent_widget = QProgressBar(self.content)
        self.main_layout.addWidget(self.loading_percent_widget)
        self.set_loading_percent(100)

        self.content.setLayout(self.main_layout)
        self.setCentralWidget(self.content)

        self.setGeometry(100, 100, 800, 800)

    def enter_search_box(self):
        self.search_keyword_action()

    def file_add_action(self):
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Photo",
            "",
            "Images (*.bmp *.pbm *.pgm *.gif *.sr *.ras *.jpeg *.jpg *.jpe *.jp2 *.tiff *.tif *.png *.mp4)",
            options=options,
        )
        self.file_add_image_db(path, commit=True)

    def file_add_image_db(self, path, commit=False):
        image = cv2.imread(path)
        if image is not None and not image.any():
            print(f"No image found : {path}")
            return
        image_data = self.image_analyzation.getImageData(
            image,
            classesData=True,
            imageFeatures=True,
            objectsFeatures=True,
            returnOriginalImage=False,
            classesConfidence=0.35,
        )
        image_data.orgImage = path
        self.img_db.addImage(image_data, commit_flag=commit)

    def file_add_folder_action(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder", "", options=options
        )
        if folder_path:
            self.selected_folder_path = folder_path
            self.setCursor(Qt.WaitCursor)
            self.index_worker = IndexFunction(
                self.image_analyzation,
                self.img_db,
                os.cpu_count(),
                folder_path,
                2,
                self.desc,
                self.vec,
            )
            self.index_worker.progress.connect(self.set_loading_percent)
            self.index_worker.done.connect(self.set_cursor_arrow)
            self.index_worker.start()
            self.set_loading_percent(0)

    def set_cursor_arrow(self):
        self.setCursor(Qt.ArrowCursor)

    def search_keyword_action(self):
        self.search_image.hide()
        text = self.search_box.text().lower()
        # text_words = text.split(" ")
        # text_words = list(filter(lambda x: x in dbf.model_names, text_words))
        self.img_db.open_connection()
        imgs = self.img_db.search_by_caption(self.vec.infer_vector(text))
        self.img_db.close_connection()

        imgs_display = DisplayList()
        for img in imgs:
            imgs_display.append(DisplayItem(img.orgImage, 0, img))

        self.set_loading_percent(0)

        self.image_add_thread = QThread()
        self.image_grid.add_images_mt(
            list(map(lambda x: x.image_data, imgs_display.items))
        )

        print(self.search_box.text())

    def search_image_action(self):
        dialog = SearchImageDialog(self.image_analyzation)
        dialog.exec()
        search_params = dialog.search_params
        if not dialog.has_image:
            return

        self.setCursor(Qt.WaitCursor)
        img_data: ImageData = search_params.data
        img_data.description = self.desc.caption(img_data.orgImage)
        img_data.vector = self.vec.infer_vector(img_data.description)
        self.search_image.show()
        self.search_image.showImage(
            imagePath=search_params.imagePath,
            img_data=img_data,
            index=search_params.selectedIndex,
        )
        # TODO : Add logic to check if the file is in the database, and if it is not it adds it to the database
        # self.file_add_image_db(search_params.imagePath)
        # Maybe if there is not a 100% match add the image to the database

        self.img_db.open_connection()
        imgs = self.img_db.search_by_image(
            [x.className for x in img_data.classes]
            if search_params.selectedIndex is None
            else [
                img_data.classes[search_params.selectedIndex].className,
            ]
        )  # sve slike sa tom odrednjemo klasom
        self.img_db.close_connection()

        image_list = DisplayList()
        # print(search_params.get_dict())
        self.set_loading_percent(0)
        for img in imgs:
            conf = self.image_analyzation.compareImages(
                imgData1=img_data,
                imgData2=img,
                compareObjects=search_params.compareObjects,
                compareWholeImages=search_params.compareWholeImages,
                maxWeightReduction=search_params.maxWeightReduction,
                containSameObjects=search_params.containSameObjects,
                confidenceCalculation=search_params.magnitudeCalculation,
                magnitudeCalculation=search_params.magnitudeCalculation,
                minObjConf=search_params.minObjConf,
                minObjWeight=search_params.minObjWeight,
                selectedIndex=search_params.selectedIndex,
            )

            if search_params.textContext:
                comp_2 = self.vec.compare_vectors(img_data.vector, img.vector)
                conf = (1 - search_params.textContextWeight) * conf + search_params.textContextWeight * comp_2
            img.similarity = conf
            image_list.append(DisplayItem(img.orgImage, conf, img))

        av = image_list.average()
        image_list.filter_sort(av)

        self.image_grid.add_images_mt(
            list(map(lambda x: x.image_data, image_list.items))
        )

        self.setCursor(Qt.ArrowCursor)

    def set_loading_percent(self, percent):
        if percent == 0:
            self.loading_percent_widget.show()
        if percent >= 100:
            self.loading_percent_widget.hide()
        self.loading_percent_widget.setValue(int(percent))

    def settings_change(self):
        dialog = SearchImageDialog(None, options_only=True)
        dialog.exec()

    def file_exit_action(self):
        QApplication.quit()

    def start_video_player(self, video_path, data, frame):
        # print(data)
        if self.video_player:
            self.video_player.closePlayer()
            self.video_player.deleteLater()
        self.video_player = VideoPlayer(fileName=video_path, data=data, start_frame = frame)
        self.video_player.setWindowTitle("Player")
        self.video_player.resize(800, 800)
        self.video_player.show()
