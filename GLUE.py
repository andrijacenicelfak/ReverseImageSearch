import sys
import threading
from time import sleep
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QWidget,
    QListWidget,
    QScrollArea,
    )
from PyQt5.QtCore import (
    Qt,
    QUrl
)
from PyQt5.QtGui import (
    QPixmap,   
)
class GUI(QMainWindow):
    def __init__(self,img_db,img_proc):
        super().__init__()
        self.initUI()
        self.show()
        self.selected_photo_path=None
        self.selected_folder_path=None
        self.img_db=img_db
        self.img_process=img_proc
        
    def initUI(self):
        self.setWindowTitle("Light")
        self.setGeometry(100,100,1000,600)
        
        main_layout=QVBoxLayout()
        
        self.selected_photo(main_layout)
        
        self.search_results(main_layout)
        
        self.buttons_layout=QHBoxLayout()
        
        self.buttons_layout.addWidget(self.btn_select_photo())
        self.buttons_layout.addWidget(self.btn_select_folder())
        
        main_layout.addLayout(self.buttons_layout)
        
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def index_folder(self,path,img_db):
        sleep(10)
        #odi do path-a
        #konekcija sa bazom
        #labela
        #ucitavaj po batch slika
        #izvuci klasifikacije i deskriptore
        #upisi u bazu
        #jmp labela
        #enableuj button
        self.btn_folder.setEnabled(True)
        pass
    
    def open_folder_dialog(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "", options=options)
        if folder_path:
            self.selected_folder_path= folder_path
            self.btn_folder.setEnabled(False)
            handle=threading.Thread(target=self.index_folder,args=(folder_path,self.img_db))
            handle.start()
            print(f"Indexed folder:{folder_path}")
    
    def btn_select_folder(self):
        self.btn_folder=QPushButton("Select Folder To Index", self)
        self.btn_folder.clicked.connect(self.open_folder_dialog)
        return self.btn_folder
    
    def display_selected_photo(self):
        pixmap = QPixmap(self.selected_photo_path)
        self.photo_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
    
    def open_image_from_search_result(self):
        pixmap = QPixmap(self.selected_photo_path)
        self.photo_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
    
    def search_results(self,main_layout):
        self.search_results_list = QListWidget(self)
        self.search_results_list.setIconSize(QPixmap(200,200).size())
        self.search_results_list.itemClicked.connect(self.open_image_from_search_result)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.search_results_list)
        main_layout.addWidget(scroll_area)
    
    def selected_photo(self,main_layout):
        self.photo_label=QLabel(self)
        self.photo_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.photo_label)
    
    def btn_select_photo(self):
        self.btn_photo=QPushButton("Select Photo", self)
        self.btn_photo.clicked.connect(self.open_photo_dialog)
        return self.btn_photo
        
    def open_photo_dialog(self):
        options=QFileDialog.Options()
        photo_path,_=QFileDialog.getOpenFileName(self, "Select Photo", "","Images (*.bmp *.pbm *.pgm *.gif *.sr *.ras *.jpeg *.jpg *.jpe *.jp2 *.tiff *.tif *.png)", options=options)
        if photo_path:
            self.selected_photo_path=photo_path
            self.display_selected_photo()
            self.btn_photo.setEnabled(False)
            handle=threading.Thread(target=self.display_results,args=(photo_path,))
            handle.start()
    def display_results(self,photo_path):
        #iskoristi andrijino za detekciju i poredjivanje
        #vrati rezultate i accuracy za svaki rezultat u batchevima
        #displayuj u batchevima
        sleep(10)
        self.btn_photo.setEnabled(True)
        pass