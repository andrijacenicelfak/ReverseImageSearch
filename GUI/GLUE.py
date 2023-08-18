from functools import reduce
import copy
import multiprocessing as mp
import os
import time
import cv2
import sys
from GUI.VideoPlayerFile_old import VideoPlayer
#sys.path.append(r"C:\dev\Demo\Video")
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
from Video.histoDThresh import  summ_video_parallel,FrameData

class DisplayItem:
    def __init__(self,image_path,accuracy):
        self.image_path=image_path
        self.accuracy=accuracy

class DisplayList:

    def __init__(self):
        self.items=[]

    def __iter__(self):
        return iter(self.items)
    
    def append(self,item):
        self.items.append(item)

    def filter_sort(self,val):
        self.items=[item for item in self.items if item.accuracy>=val]
        self.items.sort(key=lambda item: item.accuracy,reverse=True)       
 
    def clear(self):
        self.items.clear()

    def average(self):
        suma = 0
        if len(self.items) == 0:
            return 0
        maxel = max(self.items, key=lambda a: a.accuracy)
        for i in self.items:
            suma += i.accuracy
        suma -= maxel.accuracy
        return suma / max(1, len(self.items))

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
        self.image_list=DisplayList()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.thread_manager=MyThreadManager()
        self.num_of_processes=mp.cpu_count()
        self.pool=mp.Pool(self.num_of_processes) 
        self.video_player = None      
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
        xD=time.time()
        img_db.open_connection()
        
        with mp.Manager() as manager:
            for batch in search2(path):
                for img_path in batch:
                    queue=manager.Queue() 
                    if '.mp4' not in img_path:
                        print(f"File path IMAGE:{img_path}")
                        image=cv2.imread(img_path)
                        image_data = self.img_process.getImageData(image, classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = False, classesConfidence=0.35)
                        image_data.orgImage = img_path
                        img_db.addImage(image_data,commit_flag=False)
                    else :
                        print(f"File path VIDEO:{img_path}")
                        modelsum=0
                        summ_video_parallel(img_path,queue,self.num_of_processes,self.pool)
                        i=0
                        os.makedirs("C:\\kf3", exist_ok=True) ##x
                        while i != self.num_of_processes:
                            frame_data=queue.get()
                            if frame_data is None:
                                i+=1
                            else:
                                video_name = os.path.basename(img_path)##x
                                image_data = self.img_process.getImageData(frame_data.frame, classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = False, classesConfidence=0.35)
                                fake_image_path = f"C:\\kf3\\{video_name}\\{str(frame_data.frame_number)}.png"##x
                                real_video_path_plus_image = img_path + f"\\{str(frame_data.frame_number)}.png"
                                os.makedirs(f"C:\\kf3\\{video_name}", exist_ok=True)##x
                                cv2.imwrite(fake_image_path, frame_data.frame)##x
                                
                                image_data.orgImage=real_video_path_plus_image##x
                                
                                img_db.addImage(image_data,commit_flag=False)
        # self.pool.close()
        # self.pool.join()
        img_db.commit_changes()
        img_db.close_connection()
        # print(f"Model:{modelsum}") 
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
        self.image_list.clear()
        self.search_results_list.clear()
        img = cv2.imread(photo_path)
        image_data = self.img_process.getImageData(img ,classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = False, classesConfidence=0.35)
        print(list(map(lambda x: x.className, image_data.classes)))
        img_db.open_connection()
        imgs = img_db.search_by_image([ x.className for x in image_data.classes])#sve slike sa tom odrednjemo klasom
        img_db.close_connection()

        length=len(imgs)
        sum=0
        for img in imgs:
            start=time.perf_counter()
            confidence=self.img_process.compareImages(imgData1=image_data,imgData2=img,compareObjects=True,compareWholeImages = True)
            sum+=time.perf_counter()-start
            self.image_list.append(DisplayItem(img.orgImage, confidence))
        
        
        average = self.image_list.average()
        self.image_list.filter_sort(average)
        
        print(f"Compare time:{sum}")
        print(f"Average time per image:{sum/max(1, length)}")
        print(f"Number of images:{length}")
            
        for item in self.image_list:
            
            self.add_image_to_grid(item.image_path)
            # self.update()

        self.setCursor(Qt.ArrowCursor)
        self.btn_photo.setEnabled(True)
        
        print(f"Total:{time.time()-xD}")
        
    def add_image_to_grid(self, image_path:str):
        
        paths = image_path.split('\\')
        
        if paths[-2].lower().endswith(('.mp4',)):
            not_video_path =  f"C:\\kf3\\{paths[-2]}\\{paths[-1]}"
        
        pixmap = QPixmap(not_video_path)
        
        icon = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item = QListWidgetItem(QIcon(icon), "")
        item.setData(Qt.UserRole, image_path)
        self.search_results_list.addItem(item)

    def open_selected_image(self, item):
        image_path = item.data(Qt.UserRole)
        print(image_path)
        paths = image_path.rsplit("\\", 1)
        if paths[0].lower().endswith(('.mp4',)):              
            print(paths)
            frame_num = paths[1].split('.')[0]
            print(frame_num)
            self.start_player(paths[0], (int(frame_num) // 30) * 1000)
            # self.thread_manager.start_thread(function=self.start_player ,args=(paths[0],))
        else:
            if image_path:
                QDesktopServices.openUrl(QUrl.fromLocalFile(image_path))    
            
    def start_player(self, video_path, position):
        self.video_player = VideoPlayer(fileName=video_path)
        self.video_player.setWindowTitle("Player")
        self.video_player.resize(600, 400)
        self.video_player.mediaPlayer.setPosition(position)
        self.video_player.show() 
        # time.sleep(100)     
def search2(startDirectory):
    file_list = os.listdir(startDirectory)
    yield_this=[]
    counter=0
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.mp4')):
            startDirectory = os.path.normpath(startDirectory)
            image_path = os.path.join(startDirectory , file_name)
            
            if image_path.endswith('.mp4'):
                yield [image_path]
            else:
                yield_this.append(image_path)
                counter+=1
                    
                if counter==128:
                    counter=0
                    yield yield_this
                    yield_this.clear()
            
            if yield_this:
                yield yield_this