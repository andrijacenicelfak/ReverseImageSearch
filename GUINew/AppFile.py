from functools import reduce
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from DB.SqliteDB import ImageDB
import DB.Functions as dbf
from GUINew.DisplayFile import *
from GUINew.ImageGridFile import ImageGrid
from GUINew.SearchImageDialogFile import SearchImageDialog
from GUINew.SearchImageFile import *

from ImageAnalyzationModule.ImageAnalyzationFile import *
from GUI.GUIFunctions import *


class App(QMainWindow):
    def __init__(self, image_analyzation : ImageAnalyzation, img_db : ImageDB):
        super().__init__()
        self.setWindowTitle("App")
        self.content = QWidget()
        self.image_analyzation = image_analyzation
        self.img_db = img_db
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

        self.setGeometry(200, 200, 1200, 1200)

        # self.search_image.hide()
        # self.image_grid.hide()
        
    def file_add_action(self):
        #TODO
        print("add")

    def file_add_folder_action(self):
        #TODO
        print("folder")
    
    def search_keyword_action(self):
        self.search_image.hide()
        text = self.search_box.text()
        text_words = text.split(' ')
        text_words = list(filter(lambda x : x in dbf.model_names, text_words))
        self.img_db.open_connection()
        imgs = self.img_db.search_by_image(text_words)
        self.img_db.close_connection()

        imgs_display = DisplayList()
        for img in imgs:
            imgs_display.append(DisplayItem(img.orgImage, 0, img))

        self.image_grid.addImages(list(map(lambda x : x.image_data,imgs_display.items)))

        print(self.search_box.text())

    def search_image_action(self):
        dialog = SearchImageDialog(self.image_analyzation)
        dialog.exec()
        search_params = dialog.search_params
        if not dialog.has_image:
            return
        
        self.setCursor(Qt.WaitCursor)
        img_data : ImageData = search_params.data
        self.search_image.show()
        self.search_image.showImage(imagePath=search_params.imagePath, img_data=img_data)
        
        self.img_db.open_connection()
        print(search_params.selectedIndex)
        imgs = self.img_db.search_by_image([ x.className for x in img_data.classes] if search_params.selectedIndex is None else  [img_data.classes[search_params.selectedIndex].className,])#sve slike sa tom odrednjemo klasom
        self.img_db.close_connection()

        image_list = DisplayList()

        for img in imgs:
            if img.orgImage == "C:\\Users\\best_intern\\Downloads\\val2\\000000104424.jpg":
                print("HERE")
            conf = self.image_analyzation.compareImages(
                imgData1= img_data,
                imgData2= img,
                compareObjects= search_params.compareObjects,
                compareWholeImages= search_params.compareWholeImages,
                maxWeightReduction= search_params.maxWeightReduction,
                containSameObjects= search_params.containSameObjects,
                confidenceCalculation= search_params.magnitudeCalculation,
                magnitudeCalculation= search_params.magnitudeCalculation,
                minObjConf= search_params.minObjConf,
                minObjWeight= search_params.minObjWeight,
                selectedIndex= search_params.selectedIndex
            )
            image_list.append(DisplayItem(img.orgImage, conf, img))
        
        av = image_list.average()
        image_list.filter_sort(av)

        self.image_grid.addImages(list(map(lambda x : x.image_data,image_list.items)))

        self.setCursor(Qt.ArrowCursor)


    def settings_change(self):
        dialog = SearchImageDialog(None, options_only=True)
        dialog.exec()

    def file_exit_action(self):
        QApplication.quit()