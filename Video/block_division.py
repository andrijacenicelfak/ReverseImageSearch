import cv2
import multiprocessing as mp
import time
import numpy as np
from numba import jit

@jit(nopython=True)
def divide_matrix_into_blocks(matrix1, matrix2, x):
    rows1, cols1, _ = matrix1.shape
    rows2, cols2, _ = matrix2.shape

    block1_rows = rows1 // x
    block1_cols = cols1 // x

    block2_rows = rows2 // x
    block2_cols = cols2 // x

    average_value1 = np.mean(matrix1)
    block_results1 = np.zeros((x, x), dtype=np.int32)

    average_value2 = np.mean(matrix2)
    block_results2 = np.zeros((x, x), dtype=np.int32)

    for i in range(x):
        for j in range(x):
            block1 = matrix1[i * block1_rows:(i + 1) * block1_rows, j * block1_cols:(j + 1) * block1_cols]
            block1_sum = np.mean(block1)
            block_results1[i, j] = 1 if block1_sum > average_value1 else 0

    for i in range(x):
        for j in range(x):
            block2 = matrix2[i * block2_rows:(i + 1) * block2_rows, j * block2_cols:(j + 1) * block2_cols]
            block2_sum = np.mean(block2)
            block_results2[i, j] = 1 if block2_sum > average_value2 else 0

    return np.sum(np.bitwise_xor(block_results1.flatten(), block_results2.flatten()))

def videoTest(video_path, start_frame, end_frame, output_queue):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    
    prev_frame = None
    prev_sim = 0
    frame_counter=-1
    for x in range(start_frame, end_frame):

        ret, frame = cap.read()
        frame_counter+=1
        if ret and prev_frame is not None:
                similarity=divide_matrix_into_blocks(frame,prev_frame,4)
                if similarity > 4 :
                    if prev_sim < 4:
                        cv2.imwrite(f"kf3/{start_frame+x}.jpg", frame)
                        frame_counter=0
                if  frame_counter==30:
                    cv2.imwrite(f"kf3/{start_frame+x}.jpg", frame)
                    frame_counter=0
                prev_sim = similarity
        prev_frame = frame.copy()
    cap.release()
    cv2.destroyAllWindows()
def parallelVideoTest(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Total frames:{total_frames}")
    num_cores = mp.cpu_count() 
    frames_per_core = total_frames // num_cores
    processes = []
    for i in range(0, num_cores):
        start_frame = i*frames_per_core
        end_frame = start_frame + frames_per_core if i < num_cores - 1 else total_frames
        process = mp.Process(target=videoTest, args=(video_path, start_frame, end_frame, mp.Queue()))
        processes.append(process)
        process.start()
        
    for proc in processes:
        proc.join()

if __name__ == "__main__":
    
    startTime = time.time()
    parallelVideoTest('vid.mp4')
    endTime = time.time()
    elapsed_time = (endTime - startTime)
    print(f"Elapsed time: {elapsed_time:.4f} seconds")