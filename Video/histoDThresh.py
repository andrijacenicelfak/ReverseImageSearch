import cv2
import multiprocessing as mp

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
    output_queue.put(None)

def summ_video_parallel(video_path:str,queue,processes):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    num_cores = mp.cpu_count()
    frames_per_core = total_frames // num_cores 
    print("Nesto se desava")
    for i in range(0, num_cores):
        start_frame = i*frames_per_core
        end_frame = start_frame + frames_per_core if i < num_cores - 1 else total_frames
        process = mp.Process(target=summ_video, args=(video_path, start_frame, end_frame,queue))
        processes.append(process)
        process.start()