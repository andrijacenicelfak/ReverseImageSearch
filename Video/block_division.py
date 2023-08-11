import os
import cv2
import multiprocessing as mp
import time
import numpy as np
from numba import jit

def clean_dir(path):
    try:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  
        print(f"Directory '{path}' has been cleaned.")
    except OSError as e:
        print(f"Error cleaning directory: {e}")

@jit(nopython=True)
def divide_matrix_into_blocks(matrix1, matrix2, x):
    rows1, cols1, _ = matrix1.shape
    block1_rows = rows1 // x
    block1_cols = cols1 // x
    average_value1 = np.mean(matrix1)

    block_results = np.zeros((x, x), dtype=np.int32)

    for i in range(x):
        for j in range(x):
            block1_sum = np.mean(matrix1[i * block1_rows:(i + 1) * block1_rows, j * block1_cols:(j + 1) * block1_cols])
            block2_sum = np.mean(matrix2[i * block1_rows:(i + 1) * block1_rows, j * block1_cols:(j + 1) * block1_cols])
            block_results[i, j] = 1 if block1_sum > average_value1 or block2_sum > average_value1 else 0
    
    return np.sum(block_results)

def videoTest(video_path, start_frame, end_frame, output_queue):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    prev_frame = None
    prev_sim = 0
    frame_counter = -1
    
    for x in range(start_frame, end_frame):
        ret, frame = cap.read()
        frame_counter += 1
        
        if ret and prev_frame is not None:
            similarity = divide_matrix_into_blocks(frame, prev_frame, 4)
            if similarity > 9:
                if prev_sim < 9 or frame_counter == 30:
                    output_queue.put((start_frame + x, frame))
            
            prev_sim = similarity
        
        prev_frame = frame
    
    cap.release()

def parallelVideoTest(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Total frames: {total_frames}")
    
    num_cores = mp.cpu_count() 
    frames_per_core = total_frames // num_cores
    
    manager = mp.Manager()
    output_queue = manager.Queue()
    
    processes = []
    for i in range(0, num_cores):
        start_frame = i * frames_per_core
        end_frame = start_frame + frames_per_core if i < num_cores - 1 else total_frames
        
        process = mp.Process(target=videoTest, args=(video_path, start_frame, end_frame, output_queue))
        processes.append(process)
        process.start()
        
    for proc in processes:
        proc.join()
    
    # Batch writing frames
    frame_batch = []
    while not output_queue.empty():
        frame_idx, frame = output_queue.get()
        frame_batch.append((frame_idx, frame))
    
    if frame_batch:
        frame_batch.sort(key=lambda x: x[0])
        for frame_idx, frame in frame_batch:
            cv2.imwrite(f"kf4/{frame_idx}.jpg", frame)

if __name__ == "__main__":
    clean_dir('kf4')
    
    startTime = time.time()
    
    parallelVideoTest('vid.mp4')
    
    endTime = time.time()
    
    elapsed_time = (endTime - startTime)
    
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    