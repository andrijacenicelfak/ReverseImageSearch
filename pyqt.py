import sys, time, cv2,threading, numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QListWidget, QListWidgetItem, QWidget,QHBoxLayout, QScrollArea
from PyQt5.QtGui import QPixmap, QIcon,QDesktopServices
from PyQt5.QtCore import Qt,QUrl

from ImageAnalyzation import ImageAnalyzation
from index import read_images_from_dir
from sqliteDB import ImageDB,DBStruct


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
        self.list.append(struct)
    def contains(self,image_path):
        for struct in self.list:
            if struct.image_path==image_path:
                return struct
        return None
    def sort(self):
        self.list.sort(key=lambda x: x.accuracy, reverse=True)

class ReverseImageSearch(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.imageDB = ImageDB()
        self.ia = ImageAnalyzation("yolov8s.pt", "cpu")
        self.selected_folder_path = "C:\\Users\\mdsp\\Pictures\\"
        self.selected_photo_path = None

    def initUI(self):
        self.setWindowTitle("Reverse Image Search")
        self.setGeometry(100, 100, 1000, 600)
        
        main_layout = QVBoxLayout()

        self.photo_label = QLabel(self)
        self.photo_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.photo_label)

        self.search_results_list = QListWidget(self)
        self.search_results_list.setIconSize(QPixmap(150, 150).size())
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.search_results_list)
        main_layout.addWidget(scroll_area)

        self.select_photo_button = QPushButton("Select Photo", self)
        self.select_photo_button.clicked.connect(self.open_photo_dialog)

        self.select_folder_button = QPushButton("Select Folder To Index", self)
        self.select_folder_button.clicked.connect(self.open_folder_dialog)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.select_photo_button)
        buttons_layout.addWidget(self.select_folder_button)

        main_layout.addLayout(buttons_layout)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.search_results_list.itemClicked.connect(self.open_image_from_search_result)
        self.selected_photo_path = ""

    def open_photo_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Photo", "", "Images (*.png *.jpg *.jpeg *.webp)", options=options)
        if file_path:
            self.selected_photo_path = file_path
            self.display_photo()
            self.display_search_results()

    def open_folder_dialog(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "", options=options)
        if folder_path:
            self.selected_folder_path = folder_path
            # threading.Thread(target=self.index_path,args=(self.selected_folder_path,self.imageDB)).start() #TODO: concurrentqueue/stack
            self.index_path(folder_path)
    
    def display_photo(self):
        pixmap = QPixmap(self.selected_photo_path)
        self.photo_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
    
    def open_image_from_search_result(self, item):
        path_to_image = item.data(Qt.UserRole)
        if path_to_image:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path_to_image))

    def display_search_results(self):
        self.search_results_list.clear()
        if self.selected_photo_path:
            search_results = self.get_search_results()
            search_results.sort()
            for result in search_results:
                pixmap = QPixmap(result.image_path)
                icon = pixmap.scaled(100, 100, Qt.KeepAspectRatio)
                item = QListWidgetItem(QIcon(icon), f"Accuracy: {result.accuracy}%")
                item.setData(Qt.UserRole, result.image_path)
                self.search_results_list.addItem(item)
    
    def index_path(self,path):
        start_time=time.time()
        orb=cv2.ORB_create()
        i=0
        for (img_name, image) in read_images_from_dir(path):
            data = (self.ia.getImageData(image))#TODO: napravi u iterator pa preko yield lol
            for d in data:
                box = image[d[1][1]:d[1][3], d[1][0]:d[1][2]]#TODO: napravi da lepo radi sa koleginim structom D:D
                _, desc = orb.detectAndCompute(box, None)
                self.imageDB.addImage(DBStruct(d[0],img_name,desc))
                i+=1
        end_time=time.time()-start_time
        print(f"Images Indexed:{i} in {end_time/60} minutes")
    
    def get_search_results(self,):
        orb=cv2.ORB_create()
        img=cv2.imread(self.selected_photo_path,cv2.COLOR_BGR2GRAY)#izabrana slika
        terms=self.ia.getImageData(img)#('kuce',koordinate na slici)
        display_list=DisplayList()
        matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        for term in terms:
            structs=self.imageDB.searchImageByTerm(term[0])
            detected_object=img[term[1][1]:term[1][3], term[1][0]:term[1][2]]
            _,descriptor=orb.detectAndCompute(detected_object,None)
            for struct in structs:
                if descriptor is None or struct.descriptor is None:
                    break
                matches=matcher.match(descriptor.astype(np.uint8),struct.descriptor.astype(np.uint8))
                accuracy=(len(matches)/len(descriptor))*100
                if accuracy>15:
                    result=display_list.contains(struct.path_to_image)
                    if result is None:
                        display_list.append(DisplayStruct(struct.path_to_image,accuracy))
                    else:
                        result.accuracy+=accuracy
                        result.accuracy/=2
                        #TODO: resi ovo, realno je glupo...
            #isecem deo moje slike da poredim sa deskriptorima koje dobijam iz iste klase
        return display_list

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ReverseImageSearch()
    window.show()
    sys.exit(app.exec_())
