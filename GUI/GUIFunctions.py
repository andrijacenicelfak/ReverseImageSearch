import numpy as np
from ImageAnalyzationModule.ImageAnalyzationFile import ImageData, ImageClassificationData, BoundingBox
import cv2
import random
from PyQt5.QtGui import QPixmap, QImage

random.seed(151581818)
colors = [tuple(int(random.randrange(255)) for _ in range(3)) for _ in range(50)]

#Draw all rectangles or draw only selected indexes or draw one
def drawClasses(imgData : ImageData, img, fontSize = 2.4, index = None):
    print(f"index : {index}")
    if index is None:
        for i, oclass in enumerate((imgData.classes)):
            drawBoundingBox(image=img, boundingBox= oclass.boundingBox, index=i, className=oclass.className, fontSize=fontSize)
    elif isinstance(index, list):
        for i in index:
            if i < 0 or i > len(imgData.classes):
                continue
        drawBoundingBox(image=img, boundingBox= imgData.classes[i].boundingBox, index=i, className=imgData.classes[i].className, fontSize=fontSize)

    elif isinstance(index, int):
        if index < 0 or index > len(imgData.classes):
            return img
        drawBoundingBox(image=img, boundingBox= imgData.classes[index].boundingBox, index=index, className=imgData.classes[index].className, fontSize=fontSize)
    return img

def drawBoundingBox(image, boundingBox : BoundingBox, index : int, className : str, fontSize = 2.4):
        cv2.rectangle(image, (boundingBox.x1, boundingBox.y1), (boundingBox.x2, boundingBox.y2), thickness=4, lineType=cv2.LINE_AA, color=colors[index])
        cv2.rectangle(image, (boundingBox.x1, boundingBox.y1), (boundingBox.x2, boundingBox.y1 + int(fontSize*15)), thickness=-1, lineType=cv2.LINE_AA, color=colors[index])
        cv2.putText(image, className, (boundingBox.x1, boundingBox.y1 + int(fontSize*15)), 1, fontSize, (255,255,255), int(fontSize//4+1), cv2.LINE_AA)

def resizeImage(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def numpy_to_pixmap(numpy_image):
    height, width, channel = numpy_image.shape
    bytes_per_line = 3 * width
    q_image = QImage(numpy_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_image)
    return pixmap

#if video return real video path with tuple(video path, frame_num), otherwise image path
def format_if_video_path(path)->tuple:
    paths = path.rsplit("\\", 1)
    if paths[0].lower().endswith(('.mp4',)):
        frame_num = paths[1].split('.')[0]
        return (paths[0], frame_num)
    return (path, None)             


#returns path of  either image or video frame 
def format_image_path(path):
    paths = path.split('\\')
    new_path = path 

    #PROMENI U APPDDATa
    TEMP_VIDEO_FILE_PATH = "C:\\kf3\\"
    
    if len(paths) > 1:
        if paths[-2].lower().endswith(('.mp4',)):
            new_path =  TEMP_VIDEO_FILE_PATH + f"{paths[-2]}\\{paths[-1]}"
    
    return new_path