import threading
from FileExplorer import FileExplorer
from time import sleep
import cv2
from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QWidget,
    QListWidgetItem,
    QListWidget,
    QScrollArea,
    )
from PyQt5.QtCore import (
    Qt,
    QUrl,
)
from PyQt5.QtGui import (
    QPixmap,
    QDesktopServices,   
    QIcon,
)
from ImageAnalyzation import ImageClassificationData, ImageData

class DisplayStruct:
    def __init__(self,image_path,accuracy):
        self.image_path=image_path
        self.accuracy=accuracy

class DisplayList:
    def __init__(self):
        self.list=[]
    def __iter__(self):
        return iter(self.list)
    def append(self,struct):
        existing_struct = self.contains(struct.image_path)
        if existing_struct:
            existing_struct.accuracy = (existing_struct.accuracy + struct.accuracy) / 2
        else:
            self.list.append(struct)
    def contains(self,image_path):
        for struct in self.list:
            if struct.image_path==image_path:
                return struct
        return None
    def sort(self):
        self.list.sort(key=lambda x: x.accuracy, reverse=True)

from SqliteDB import  ImageDB
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
    

    def index_folder(self,path,img_db:ImageDB):
        print("Hello, orlDW!") #DO NOT DELETE EVERYTHING BREAKS WITHOUT THIS PRINT
        file_exp = FileExplorer(path)
        img_db.open_connection()
        for batch in file_exp.search2():
            for (img_path, image) in batch:
                image_data = self.img_process.getImageData(image, True, True, True)
                image_data.orgImage = img_path
                img_db.addImage(image_data)

        img_db.close()  
        self.setCursor(Qt.ArrowCursor)
        self.btn_folder.setEnabled(True)
        
    
    def open_folder_dialog(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "", options=options)
        if folder_path:
            self.selected_folder_path= folder_path
            self.btn_folder.setEnabled(False)
            self.setCursor(Qt.WaitCursor)
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
    
    def search_results(self,main_layout):
        self.search_results_list = QListWidget(self)
        self.search_results_list.setIconSize(QPixmap(200,200).size())
        self.search_results_list.itemClicked.connect(self.open_selected_image)
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
            self.setCursor(Qt.WaitCursor)
            self.display_selected_photo()
            self.btn_photo.setEnabled(False)
            handle=threading.Thread(target=self.display_results,args=(photo_path,self.img_db))
            handle.start()
    def display_results(self,photo_path,img_db:ImageDB):
        img_list=DisplayList()
        image_data = self.img_process.getImageData(cv2.imread(photo_path), True, False, True)
        img_db.open_connection()
        for obj in image_data.classes:
            imgs = img_db.searchImageByTerm(obj.className)#sve slike sa tom odrednjemo klasom
            for img in imgs: 
                confidence = self.img_process.compareImageClassificationData(ImageClassificationData(img.term, None, img.descriptor), ImageClassificationData(obj.className, None, obj.features), None, None, False, False)
                img_list.append(DisplayStruct(img.path_to_image,confidence))
        
        img_list.sort()
        for struct in img_list:
            self.add_image_to_grid(struct.image_path,struct.accuracy)
        
        self.setCursor(Qt.ArrowCursor)
        img_db.close()
        
        self.btn_photo.setEnabled(True)
    def add_image_to_grid(self, image_path,accuracy):
        pixmap = QPixmap(image_path)
        icon = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item = QListWidgetItem(QIcon(icon), f"Accuracy: {round(accuracy*100,2)}%")
        item.setData(Qt.UserRole, image_path)
        accuracy_label = QLabel(f"Accuracy: {accuracy:.2f}")
        accuracy_label.setAlignment(Qt.AlignCenter)
        self.search_results_list.addItem(item)
    
    def open_selected_image(self, item):
        image_path = item.data(Qt.UserRole)
        if image_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(image_path))