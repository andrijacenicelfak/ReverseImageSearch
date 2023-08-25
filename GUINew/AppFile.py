from functools import reduce
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, qInstallMessageHandler, QUrl, QRunnable, QThreadPool, QObject
from DB.SqliteDB import ImageDB
import DB.Functions as dbf
from GUINew.DisplayFile import *
from GUINew.ImageGridFile import ImageGrid
from GUINew.IndexFolderDialogFile import IndexFolderDialog

from GUINew.SearchImageDialogFile import SearchImageDialog
from GUINew.SearchImageFile import *

from ImageAnalyzationModule.ImageAnalyzationFile import *
from GUI.GUIFunctions import *
import multiprocessing as mp
from FileSystem.FileExplorerFile import search

from Video.histoDThresh import  summ_video_parallel,FrameData





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


def handle(type, context, message):
    pass



class App(QMainWindow):
    def __init__(self, image_analyzation : ImageAnalyzation, img_db : ImageDB):
        super().__init__()

        #
        self.thread_manager=MyThreadManager()
        self.num_of_processes=mp.cpu_count()
        self.pool=mp.Pool(self.num_of_processes) 
        # qInstallMessageHandler(handle)
        
        #
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

        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "", options=options)
        if folder_path:
            self.selected_folder_path= folder_path
            self.setCursor(Qt.WaitCursor)
            self.thread_manager.start_thread(function=self.index_folder,args=(folder_path,self.img_db))
 
 

     
    def index_folder(self,path,img_db):
            xD=time.time()
            img_db.open_connection()
            
            with mp.Manager() as manager:
                for batch in search(path):
                    for img_path in batch:
                        queue=manager.Queue() 
                        if '.mp4' not in img_path:
                            # print(f"File path IMAGE:{img_path}")
                            image=cv2.imread(img_path)
                            image_data = self.image_analyzation.getImageData(image, classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = False, classesConfidence=0.35)
                            image_data.orgImage = img_path
                            img_db.addImage(image_data,commit_flag=False)
                        else :
                            # print(f"File path VIDEO:{img_path}")
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
                                    image_data = self.image_analyzation.getImageData(frame_data.frame, classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = False, classesConfidence=0.35)
                                    fake_image_path = f"C:\\kf3\\{video_name}\\{str(frame_data.frame_number)}.png"##x
                                    real_video_path_plus_image = img_path + f"\\{str(frame_data.frame_number)}.png"
                                    os.makedirs(f"C:\\kf3\\{video_name}", exist_ok=True)##x
                                    cv2.imwrite(fake_image_path, frame_data.frame)##x
                                    
                                    image_data.orgImage=real_video_path_plus_image##x
                                    
                                    img_db.addImage(image_data,commit_flag=False)
            img_db.commit_changes()
            img_db.close_connection()
            print(f"Total time:{time.time()-xD}")
            self.setCursor(Qt.ArrowCursor)
 
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