from ImageAnalyzationModule.ConvolutionalModels import *
from ImageAnalyzationModule.ImageAnalyzationFile import *
from FileSystem.FileExplorerFile import FileExplorer
import time
import numpy as np
from GUI.GUIFunctions import drawClasses, resizeImage

def objectComparisonTest():
    # fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Downloads\\val2")
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Documents\\dev\\ImageClassification\\YOLOv8\\imgs")
    # fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Desktop\\New folder (2)")
    paths = fe.search()

    ai = ImageAnalyzation("yolov8l.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, aedType=AutoEncoderDecoderS, coderDecoderModel="3S-NF-29", normalization=False)
    data: list[ImageData] = []
    for p in paths:
        data.append(ai.getImageData(cv2.imread(p), classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = True))
    fullImgs = None
    for d in data:
        for cdata in d.classes:
            img = d.orgImage[cdata.boundingBox.y1 : cdata.boundingBox.y2, cdata.boundingBox.x1 : cdata.boundingBox.x2]
            allimgs = None
            for md in data:
                for mcdata in md.classes:
                    img2 = md.orgImage[mcdata.boundingBox.y1 : mcdata.boundingBox.y2, mcdata.boundingBox.x1 : mcdata.boundingBox.x2]
                    dist = ai.compareImageClassificationData(icd1=cdata, icd2=mcdata, treshhold=0, scaleDown=False, scale=(0.8, 5), magnitudeCalculation=True, classNameComparison=False)
                    print(f"{dist} : {cdata.className} : {mcdata.className}")
                    imgs = np.concatenate([cv2.resize(img, (512, 512)), cv2.resize(img2, (512, 512))], axis=1)
                    cv2.rectangle(imgs, (512, 512), (715, 490), (0,0,0), -1)
                    cv2.putText(imgs, "sim  : %1.3f" % (dist,), (512, 512), 1, 2, (255,255,255), 1, cv2.LINE_AA)
                    if allimgs is None:
                        allimgs = cv2.resize(imgs, (512, 256))
                    else:
                        allimgs = np.concatenate([allimgs, cv2.resize(imgs, (512, 256))], axis=0)
                    cv2.imshow("slika", imgs)
                    while cv2.waitKey(0) != ord(' '):
                        time.sleep(0.4)
            if fullImgs is None:
                fullImgs = allimgs
            else:
                fullImgs = np.concatenate([fullImgs, allimgs], axis=1)
    cv2.imwrite("comparison.jpg", fullImgs)

def imageComparisonTest():
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Downloads\\val2")
    paths = fe.search()
 
    ai = ImageAnalyzation("yolov8l.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, aedType=AutoEncoderDecoderM, coderDecoderModel="1M-103")
    data: list[ImageData] = []
    for p in paths:
        data.append(ai.getImageData(cv2.imread(p), classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = True, classesConfidence=0.65))

    for d1 in data:
        paths = []
        for d2 in data:
            comp = ai.compareImages(imgData1=d1, imgData2=d2, compareWholeImages=True, minObjWeight=0.0)
            if comp > 0.01:
                img = d2.orgImage.copy()
                cv2.putText(img, f"{comp}", (5, 256), 1, 2, (255,255,255), 2, cv2.LINE_4)
                paths.append((d2, comp, img))
        
        paths.sort(key=lambda x: x[1], reverse=True)
        cv2.imshow('org', cv2.resize(d1.orgImage, (512,512)))
        cv2.imshow('simmilar', np.concatenate([cv2.resize(path[2], (256,256)) for path in paths[0:min(len(paths),4)]], axis=1))
        while cv2.waitKey(0) != ord(' '):
            time.sleep(0.5)
    return

def rectDrawTest():
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Downloads\\val2")
    paths = fe.search()

    ai = ImageAnalyzation("yolov8l.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, aedType=AutoEncoderDecoderM, coderDecoderModel="1M-103")
    data: list[ImageData] = []
    for p in paths:
        data.append(ai.getImageData(cv2.imread(p), classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = True))
    
    for d in data:
        img = drawClasses(d, d.orgImage)
        cv2.imshow("img", resizeImage(img, None, 512))
        while cv2.waitKey(0) != 27:
            time.sleep(0.3)

    return
if __name__ == "__main__":
    objectComparisonTest()
    cv2.destroyAllWindows()
    print("Done")