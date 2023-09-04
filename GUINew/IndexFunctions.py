import multiprocessing as mp
import gensim
import os
import pathlib
import time
from DB.SqliteDB import ImageDB
import FileSystem.FileExplorerFile as fe
import cv2
from GUI.GUIFunctions import TEMP_VIDEO_FILE_PATH
from ImageAnalyzationModule.Describe import Describe
from ImageAnalyzationModule.ImageAnalyzationFile import ImageAnalyzation
from ImageAnalyzationModule.Vectorize import Vectorize
from Video.histoDThresh import *
from queue import Empty
from PyQt5.QtCore import pyqtSignal, QThread
VIDEO_THUMBNAIL_SIZE = 300

def image_load(input_queue : mp.Queue, files : list[str]):
    for file in files:
        ext = pathlib.Path(file).suffix
        if ext in fe.IMG_EXTENTIONS:
            img = cv2.imread(file)
            input_queue.put((img, file))
        else:
            sum_video_all(file, input_queue)
    input_queue.put(None)

def analyze_image(img_and_path, ia : ImageAnalyzation, database : ImageDB,desc:Describe,vec:Vectorize):
    image, path = img_and_path
    if image is not None and not image.any() :
        print(f"No image found : {path}")
        return
    image_data = ia.getImageData(
        image,
        classesData=True,
        imageFeatures=True,
        objectsFeatures=True,
        returnOriginalImage=False,
        classesConfidence=0.35,
    )
    image_data.description=desc.caption(image)
    image_data.vector=vec.infer_vector(image_data.description)
    image_data.orgImage = path
    database.addImage(image_data)

def analyze_frame(frame : FrameData, ia : ImageAnalyzation, database : ImageDB, save_file_queue : mp.Queue,desc:Describe,vec:Vectorize):
    video_name = os.path.basename(frame.video_path)
    # print(f"{video_name}:{frame.frame_number}")
    image_data = ia.getImageData(
        frame.frame,
        classesData=True,
        imageFeatures=True,
        objectsFeatures=True,
        returnOriginalImage=False,
        classesConfidence=0.35,
    )
    rgb_frame=cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
    image_data.description=desc.caption(rgb_frame)
    image_data.vector=vec.infer_vector(image_data.description)
    fake_image_path =  TEMP_VIDEO_FILE_PATH + f"\\{video_name}\\{str(frame.frame_number)}.png"  ##x
    real_video_path_plus_image = frame.video_path + f"\\{str(frame.frame_number)}.png"

    image_data.orgImage = real_video_path_plus_image  ##x
    save_file_queue.put((TEMP_VIDEO_FILE_PATH +f"\\{video_name}", fake_image_path, frame.frame))

    database.addImage(image_data, commit_flag=False)

def save_worker(save_queue: mp.Queue, num_of_adders : int):
    i = 0
    while True:
        try:
            data = save_queue.get()
            if data is None:
                i+=1
            else:
                os.makedirs(data[0], exist_ok=True)
                cv2.imwrite(data[1], cv2.resize(data[2], (VIDEO_THUMBNAIL_SIZE, VIDEO_THUMBNAIL_SIZE)))
            if i == num_of_adders:
                break
        except Empty:
            continue
    return

class IndexFunction(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal()
    def __init__(self, ia : ImageAnalyzation, database : ImageDB,num_of_proc : int, path : str,desc:Describe,vec:Vectorize):
        super().__init__()
        self.ia = ia
        self.database = database
        self.desc=desc
        self.vec=vec
        self.num_of_proc = num_of_proc
        self.path = path
        pass

    def analyze_files(self, input_queue: mp.Queue, num_files : int, save_file_queue : mp.Queue):
        i = 0
        num = 0
        last = 0
        self.progress.emit(0)
        file_count = num_files
        while True:
            try:
                data = input_queue.get()
                if data is None:
                    i += 1
                elif isinstance(data, FrameData):
                    file_count += 1
                    analyze_frame(data, self.ia, self.database, save_file_queue,self.desc,self.vec)
                else:
                    analyze_image(data, self.ia, self.database,self.desc,self.vec)
                if i == self.num_of_proc:
                    break
                if file_count > 0:
                    new = (98 * num/file_count)
                    if last != new:
                        self.progress.emit(int(new))
                        last = new
                num +=1
            except Empty:
                continue
        save_file_queue.put(None)
        self.progress.emit(100)

    def run(self):
        start_time = time.time()
        print(f"Indexing {self.path}")
        input_queue = mp.Queue(maxsize=256)
        save_queue = mp.Queue(maxsize=256)
        files = fe.search_all(self.path)
        num_files = len(files)
        processes = []
        os.makedirs(TEMP_VIDEO_FILE_PATH, exist_ok=True)  ##x
        start_index = 0
        chunk_size = (num_files // self.num_of_proc)
        for i in range(self.num_of_proc):
            files_chunk = files[start_index : start_index+chunk_size if i < (self.num_of_proc - 1) else (num_files)]
            start_index+= chunk_size
            p = mp.Process(target=image_load, args=(input_queue, files_chunk), name=f"Image load {i}")
            p.start()
            processes.append(p)
        self.database.open_connection()
        
        save_proc = mp.Process(target=save_worker, args=(save_queue, 1))
        save_proc.start()

        # has to analyze files in the main process because the models are on the main process
        # and model inicialization and running the models seperatly on each process is expansive
        self.analyze_files(input_queue, num_files, save_queue)
        for p in processes:
            p.join()
        save_proc.join()

        end_time = time.time()
        print("Time to index : %.2fs" % (end_time-start_time,))

        start_time = time.time()

        self.database.commit_changes()
        self.database.close_connection()

        end_time = time.time()
        print("Time to commit to db : %.2fs" % (end_time-start_time,))

        self.done.emit()
    