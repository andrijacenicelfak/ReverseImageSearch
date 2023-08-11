import sys
import os
import cv2
import multiprocessing as mp
from multiprocessing import freeze_support
# from ImageAnalyzation import ImageAnalyzation
sys.path.append(r'C:\dev\Demo\ImageAnalyzation')
sys.path.append(r'C:\dev\Demo\DB')
from ImageAnalyzation.ImageAnalyzation import AnalyzationType, ImageAnalyzation
from DB.SqliteDB import ImageDB
def clean_dir(path):
    try:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  
        print(f"Directory '{path}' has been cleaned.")
    except OSError as e:
        print(f"Error cleaning directory: {e}")
        
def summ_video(video_path, start_frame, end_frame,output_queue):


    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    prev_frame = None
    prev_hist = None
    prev_sim = 0

    threshold = 0.08
    THRESHOLD_INC = 0.01
    THRESHOLD_DEC = 0.001
    i=0
    for x in range(start_frame, end_frame):

        ret, frame = cap.read()

        if ret: 
            if i == 5:
                i=0
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                h1 = cv2.calcHist([gray_frame], [0], None, [128], [0, 256])
                
                h1 = cv2.normalize(h1, h1).flatten()

                if prev_frame is not None and prev_hist is not None:
                 
                    similarity = cv2.compareHist(h1, prev_hist, cv2.HISTCMP_BHATTACHARYYA)

                    if similarity > threshold:
                    
                        if prev_sim < threshold:
                            threshold += THRESHOLD_INC
                            output_queue.put(frame)
                            # cv2.imwrite(f"kf3/{random.randint(1,17)}.jpg", frame)
                            
            
                    else:
                        threshold -= THRESHOLD_DEC

                    prev_sim = similarity

                prev_frame = frame
                prev_hist = h1
            else:
                i+=1
        else:
            break

    cap.release()


def summ_video_parallel(video_path:str, queue:mp.Queue):

    
    cap = cv2.VideoCapture(video_path)
   
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      
    cap.release()

    num_cores = mp.cpu_count() 
    frames_per_core = total_frames // num_cores 
    processes = []
        
    for i in range(0, num_cores):
        start_frame = i*frames_per_core
        end_frame = start_frame + frames_per_core if i < num_cores - 1 else total_frames
        process = mp.Process(target=summ_video, args=(video_path, start_frame, end_frame,queue))
        processes.append(process)
        process.start()
        
    


if __name__ == "__main__":
    # clean_dir("kf3")

    # startTime = time.time()
    
    queue=mp.Queue(maxsize=0)
    processes=summ_video_parallel(r"vid.mp4C:\Users\mdsp\Desktop\videotest\vid.mp4",queue=queue)
    
    img_db=ImageDB()
    img_db.open_connection()
    img_process=ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel=r"ImageAnalyzation\ConvModelColor4R5C-28.model")
    while True:
        image=queue.get()
        if image is None:
            break
        image_data = img_process.getImageData(image, imageFeatures=True, objectsFeatures=True)
        image_data.orgImage = r"C:\Users\mdsp\Desktop\videotest\vid.mp4" 
        img_db.addImage(image_data)   
        print("MP4 ends")
        
    img_db.close_connection()
        
    for proc in processes:
        proc.join()
    
    # endTime = time.time()

    # elapsed_time = endTime - startTime
    
    # print(f"Elapsed time: {elapsed_time:.4f} seconds")