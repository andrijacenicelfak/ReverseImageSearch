from functools import reduce
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from GUINew.ImageGridFile import ImageGrid
from GUINew.SearchImageDialogFile import SearchImageDialog
from GUINew.SearchImageFile import *

from ImageAnalyzationModule.ImageAnalyzationFile import *
from GUI.GUIFunctions import *


class App(QMainWindow):
    def __init__(self, image_analyzation : ImageAnalyzation):
        super().__init__()
        self.setWindowTitle("App")
        self.content = QWidget()
        self.image_analyzation = image_analyzation
        self.setGeometry(200, 200, 1200, 1200)
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

        # self.search_image.showImage(imagePath="C:\\Users\\best_intern\\Desktop\\New folder (2)\\1.jpeg")
        # self.search_image.hide()
        # self.image_grid.hide()
        
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
        dialog = SearchImageDialog(self.image_analyzation)
        dialog.exec()
        search_params = dialog.search_params
        if search_params is None:
            return
        
        #TODO search from the database using the search params......

    def file_exit_action(self):
        QApplication.quit()