import multiprocessing as mp
import os
import time
import cv2
import sys
sys.path.append(r"C:\dev\Demo\Video")
from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QWidget,
    QListWidgetItem,
    QListWidget,
    QScrollArea,
    QGridLayout,
    )
from PyQt5.QtCore import (
    Qt,
    QUrl,
    QRunnable,
    QThreadPool,
    QObject,
)
from PyQt5.QtGui import (
    QPixmap,
    QDesktopServices,   
    QIcon,
)
sys.path.append(".\Video")
from Video.histoDThresh import  summ_video_parallel


class QDisplayListItem(QListWidgetItem):
    def __init__(self,image_path,accuracy):
        pixmap=QPixmap(image_path)
        icon=pixmap.scaled(200,200,Qt.KeepAspectRatio)
        super().__init__(QIcon(icon),"")
        self.setData(Qt.UserRole,image_path)
        self.accuracy=accuracy
        
    def __lt__(self,other):
        return self.accuracy<other.accuracy

class DisplayList:

    def __init__(self):
        self.image_paths=[]
        self.accuracies=[]
    
    def __iter__(self):
        return iter(zip(self.image_paths,self.accuracies))
    
    def append(self,image_path,accuracy):
        self.image_paths.append(image_path)
        self.accuracies.append(accuracy)
        
    def sort(self):
        temporary = sorted(zip(self.image_paths, self.accuracies), key=lambda x: x[1], reverse=True)
        for ind, (img_path, acc) in enumerate(temporary):
            self.image_paths[ind] = img_path
            self.accuracies[ind] = acc
    
    def clear(self):
        self.image_paths.clear()
        self.accuracies.clear()
        
class MyThread(QRunnable):
    
    def __init__(self,function,args):
        super().__init__()
        self.function=function
        self.args=args
        
    def run(self):
        self.function(*self.args)

class MyThreadManager(QObject):
 
    def __init__(self):
        super().__init__()
        self.thread_pool=QThreadPool()
        
    def start_thread(self,function,args):
        handle=MyThread(function=function,args=args)
        self.thread_pool.start(handle)

class GUI(QMainWindow):
    def __init__(self,img_db,img_proc):
        super().__init__()
        self.initUI()
        self.show()
        self.selected_photo_path=None
        self.selected_folder_path=None
        self.img_db=img_db
        self.img_process=img_proc
        self.img_list=DisplayList()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.thread_manager=MyThreadManager()
        
    def initUI(self):
        self.setWindowTitle("GLUEE")
        self.setGeometry(100,100,1000,600)
        
        main_layout=QVBoxLayout()
        
        self.selected_photo(main_layout)
        
        self.search_results(main_layout)
        
        # self.buttons_layout=QHBoxLayout()
        self.buttons_layout=QGridLayout()
        
        self.buttons_layout.addWidget(self.btn_select_photo())
        self.buttons_layout.addWidget(self.btn_select_folder())
        
        main_layout.addLayout(self.buttons_layout)
        
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    

    def index_folder(self,path,img_db):
        # mp.set_start_method('spawn', force=True)
        xD=time.time()
        img_db.open_connection()
        num_of_processes=mp.cpu_count()
        pool=mp.Pool(num_of_processes)
        with mp.Manager() as manager:
            for batch in search2(path):
                for img_path in batch:
                    if not img_path.endswith('.mp4'):
                        image=cv2.imread(img_path)
                        image_data = self.img_process.getImageData(image, imageFeatures=True, objectsFeatures=True)
                        image_data.orgImage = img_path
                        img_db.addImage(image_data)
                    else :
                        modelsum=0
                        queue=manager.Queue() 
                        processes=[]
                        summ_video_parallel(img_path,queue,processes,num_of_processes,pool)
                        i=0
                        while i!= num_of_processes:
                            frame=queue.get()
                            if frame is None:
                                i+=1
                            else:
                                model=time.time()
                                image_data = self.img_process.getImageData(frame, imageFeatures=True, objectsFeatures=True)
                                modelsum+=time.time()-model
                                image_data.orgImage = img_path
                                img_db.addImage(image_data,commit_flag=False)
        pool.close()
        pool.join()
        img_db.commit_changes()
        img_db.close_connection()
        print(f"Model:{modelsum}") 
        print(f"Total time:{time.time()-xD}")
        self.setCursor(Qt.ArrowCursor)
        self.btn_folder.setEnabled(True)
    
    def open_folder_dialog(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "", options=options)
        if folder_path:
            self.selected_folder_path= folder_path
            self.btn_folder.setEnabled(False)
            self.setCursor(Qt.WaitCursor)
            self.thread_manager.start_thread(function=self.index_folder,args=(folder_path,self.img_db))
    
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
        self.search_results_list.itemDoubleClicked.connect(self.open_selected_image)
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
        photo_path,_=QFileDialog.getOpenFileName(self, "Select Photo", "","Images (*.bmp *.pbm *.pgm *.gif *.sr *.ras *.jpeg *.jpg *.jpe *.jp2 *.tiff *.tif *.png *.mp4)", options=options)
        if  photo_path:
            self.selected_photo_path=photo_path
            self.setCursor(Qt.WaitCursor)
            self.display_selected_photo()
            self.btn_photo.setEnabled(False)
            self.thread_manager.start_thread(function=self.display_results,args=(photo_path,self.img_db))
    
    def display_results(self,photo_path,img_db):
        xD=time.time()
        self.img_list.clear()
        self.search_results_list.clear()
        
        img = cv2.imread(photo_path)
        image_data = self.img_process.getImageData(img ,imageFeatures=True, objectsFeatures=True)

        img_db.open_connection()
        imgs = img_db.search_by_image([ x.className for x in image_data.classes])#sve slike sa tom odrednjemo klasom
        img_db.close_connection()

        length=len(imgs)
        sum=0
        for img in imgs:
            start=time.time()
            confidence=self.img_process.compareImages(imgData1=image_data,imgData2=img,compareObjects=True,compareWholeImages = True)
            sum+=time.time()-start
            self.img_list.append(img.orgImage, confidence)
        
        print(f"Compare time:{sum}")
        print(f"Average time per image:{sum/length}")
        print(f"Number of images:{length}")
        
        self.img_list.sort()
        for index,(image_path,accuracy) in enumerate(self.img_list):
            self.add_image_to_grid(image_path,accuracy)
            self.update()
        

        self.setCursor(Qt.ArrowCursor)
        self.btn_photo.setEnabled(True)
        
        print(f"Total:{time.time()-xD}")
        
    def add_image_to_grid(self, image_path,accuracy):
        pixmap = QPixmap(image_path)
        icon = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item = QListWidgetItem(QIcon(icon),"")
        item.setData(Qt.UserRole, image_path)
        self.search_results_list.addItem(item)

    def open_selected_image(self, item):
        image_path = item.data(Qt.UserRole)
        if image_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(image_path))
            
def search2(startDirectory):
    file_list = os.listdir(startDirectory)
    yield_this=[]
    counter=0
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.mp4')):
            
            startDirectory = os.path.normpath(startDirectory)
            image_path = os.path.join(startDirectory , file_name)
            
            yield_this.append(image_path)
            counter+=1
            
            if counter==32:
                counter=0
                yield yield_this
                yield_this.clear()
    
    yield yield_this