import cv2
import multiprocessing as mp


class FrameData:
    def __init__(self, frame, frame_number, video_path):
        self.frame = frame
        self.frame_number = frame_number
        self.video_path = video_path


def summ_video(video_path, start_frame, end_frame, output_queue):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_frame = None
    prev_hist = None
    prev_sim = 0

    threshold = 0.08
    THRESHOLD_INC = 0.01
    THRESHOLD_DEC = 0.001
    i = 0
    for x in range(start_frame, end_frame):
        ret, frame = cap.read()

        if ret:
            if i == 5:
                i = 0
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                h1 = cv2.calcHist([gray_frame], [0], None, [128], [0, 256])

                h1 = cv2.normalize(h1, h1).flatten()

                if prev_frame is not None and prev_hist is not None:
                    similarity = cv2.compareHist(
                        h1, prev_hist, cv2.HISTCMP_BHATTACHARYYA
                    )

                    if similarity > threshold:
                        if prev_sim < threshold:
                            threshold += THRESHOLD_INC
                            # print(x)
                            output_queue.put(FrameData(frame=frame, frame_number=x))
                    else:
                        threshold -= THRESHOLD_DEC

                    prev_sim = similarity

                prev_frame = frame
                prev_hist = h1
            else:
                i += 1
        else:
            break

    cap.release()
    output_queue.put(None)


def sum_video_all(video_path, output_queue):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    prev_hist = None
    prev_sim = 0

    threshold = 0.08
    THRESHOLD_INC = 0.01
    THRESHOLD_DEC = 0.001
    x = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x+=1
        if x % 5 != 0:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h1 = cv2.calcHist([gray_frame], [0], None, [128], [0, 256])

        h1 = cv2.normalize(h1, h1).flatten()

        if prev_frame is not None and prev_hist is not None:
            similarity = cv2.compareHist(
                h1, prev_hist, cv2.HISTCMP_BHATTACHARYYA
            )

            if similarity > threshold:
                if prev_sim < threshold:
                    threshold += THRESHOLD_INC
                    output_queue.put(FrameData(frame=frame, frame_number=x, video_path=video_path))
            else:
                threshold -= THRESHOLD_DEC

            prev_sim = similarity

        prev_frame = frame
        prev_hist = h1
    cap.release()

def summ_video_parallel(video_path: str, queue, num_of_processes: int, pool: mp.Pool):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frames_per_core = total_frames // num_of_processes
    frame_ranges = []
    for i in range(num_of_processes):
        start_frame = i * frames_per_core
        end_frame = min((i + 1) * frames_per_core, total_frames)
        frame_ranges.append((video_path, start_frame, end_frame, queue))
    pool.starmap(summ_video, frame_ranges)
