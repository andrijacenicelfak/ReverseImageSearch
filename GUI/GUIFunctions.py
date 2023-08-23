import numpy as np
from ImageAnalyzationModule.ImageAnalyzationFile import ImageData, ImageClassificationData, BoundingBox
import cv2
import random

random.seed(151581818)
colors = [tuple(int(random.randrange(255)) for _ in range(3)) for _ in range(50)]

#Draw all rectangles or draw only selected indexes or draw one
def drawClasses(imgData : ImageData, img, fontSize = 2.4, index = None):
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
        cv2.rectangle(image, (boundingBox.x1, boundingBox.y1), (boundingBox.x2, boundingBox.y1 + fontSize*15), thickness=-1, lineType=cv2.LINE_AA, color=colors[index])
        cv2.putText(image, className, (boundingBox.x1, boundingBox.y1 + fontSize*15), fontSize//4+1, fontSize, (255,255,255), fontSize//4+1, cv2.LINE_AA)

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