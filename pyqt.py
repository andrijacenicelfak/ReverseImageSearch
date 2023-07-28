import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QListWidget, QListWidgetItem, QWidget,QScrollArea
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
import cv2

from ImageAnalyzation import ImageAnalyzation
from index import read_images_from_dir
from sqliteDB import ImageDB,DBStruct


class PhotoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.imageDB=ImageDB()
        self.ia= ImageAnalyzation("yolov8s.pt", "cuda")
        self.selected_folder_path="C:\\Users\\mdsp\\Pictures\\"
        self.selected_photo_path=None
    
    def initUI(self):
        self.setWindowTitle("Reverse Image Search")
        self.setGeometry(100, 100, 1000, 600)

        self.photo_label = QLabel(self)
        self.photo_label.setAlignment(Qt.AlignRight)

        self.search_results_list = QListWidget(self)
        self.search_results_list.setFixedSize(500, 400)
        self.search_results_list.setIconSize(QPixmap(100, 100).size())

        self.select_photo_button = QPushButton("Select Photo", self)
        self.select_photo_button.clicked.connect(self.open_photo_dialog)

        self.photo_layout = QVBoxLayout()
        self.photo_layout.addWidget(self.photo_label)

        self.select_folder_button = QPushButton("Select Folder To Index", self)
        self.select_folder_button.clicked.connect(self.open_folder_dialog)

        self.scroll_area = QScrollArea()  
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumSize(500, 400)  
        self.scroll_area.setWidget(self.search_results_list)
        self.photo_layout.addWidget(self.scroll_area)
        self.photo_layout.addWidget(self.select_photo_button)
        self.photo_layout.addWidget(self.select_folder_button)

        self.central_widget = QWidget(self)
        self.central_widget.setLayout(self.photo_layout)
        self.setCentralWidget(self.central_widget)

        self.selected_photo_path = ""

    def open_photo_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Photo", "", "Images (*.png *.jpg)", options=options)
        if file_path:
            self.selected_photo_path = file_path
            self.display_photo()
            self.display_search_results()

    def open_folder_dialog(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "", options=options)
        if folder_path:
            self.selected_folder_path = folder_path
            #index folder
            self.index_path(self.selected_folder_path)
    
    def display_photo(self):
        pixmap = QPixmap(self.selected_photo_path)
        self.photo_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def display_search_results(self):
        self.search_results_list.clear()
        if self.selected_photo_path:
            search_results = self.get_search_results()
            search_results.sort(key=lambda x:x[1],reverse=True)
            for (image_path,accuracy) in search_results:#image path,accuracy treba da bude
                    pixmap = QPixmap(image_path)
                    icon = pixmap.scaled(100, 100, Qt.KeepAspectRatio)
                    item = QListWidgetItem(QIcon(icon), f"Accuracy: {accuracy}%")#umesto sto treba accuracy da bude
                    self.search_results_list.addItem(item)
    
    def index_path(self,path):
        start_time=time.time()
        orb=cv2.ORB_create()
        i=0
        #ubacuje deskriptore i ime baze
        for (img_name, image) in read_images_from_dir(path):
            data = (self.ia.getImageData(image))#TODO: napravi u iterator pa preko yield lol
            for d in data:
                box = image[d[1][1]:d[1][3], d[1][0]:d[1][2]]
                _, desc = orb.detectAndCompute(box, None)
                self.imageDB.addImage(DBStruct(d[0],img_name,desc))
                i+=1
        end_time=time.time()-start_time
        print(f"Images Indexed:{i} in {end_time/60} minutes")
    
    def get_search_results(self,):
        orb=cv2.ORB_create()
        img=cv2.imread(self.selected_photo_path,cv2.COLOR_BGR2GRAY)#izabrana slika
        terms=self.ia.getImageData(img)#('kuce',koordinate na slici)
        image_paths=[]
        matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        for term in terms:
            structs=self.imageDB.searchImageByTerm(term[0])
            detected_object=img[term[1][1]:term[1][3], term[1][0]:term[1][2]]
            cv2.imshow("wtf",detected_object)
            _,descriptor=orb.detectAndCompute(detected_object,None)
            for struct in structs:
                if descriptor is None or struct.descriptor is None:
                    break
                matches=matcher.match(descriptor,struct.descriptor)
                accuracy=(len(matches)/len(descriptor))*100
                print(struct.path_to_image)
                if accuracy>32:
                    image_paths.append([struct.path_to_image,accuracy])
                    print(struct.path_to_image)
            #isecem deo moje slike da poredim sa deskriptorima koje dobijam iz iste klase
        return image_paths

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoApp()
    window.show()
    sys.exit(app.exec_())

    #todo: ulepsaj
    #todo: ubrzamo
    #         TODO:razilicti deskriptori, sift akaze itd posto je ovo shit
    #obavezan TODO: jedno 17 refatkora da lici na nesto
    #TODO:kada imamo sliku sa vise objekta da saberemo procente i podelimao sa 2 jer sto da ne ali 
    #     samo jednom da je prikazemo