from ImageAnalyzationModule.ImageAnalyzationFile import *
from FileSystem.FileExplorerFile import FileExplorer
import time
import numpy as np

def objectComparisonTest():
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Desktop\\New folder (2)")
    paths = fe.search()

    ai = ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="1A-27")
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
                    dist = ai.compareImageClassificationData(icd1=cdata, icd2=mcdata, treshhold=0)
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

    ai = ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="1A-27")
    data: list[ImageData] = []
    for p in paths:
        data.append(ai.getImageData(cv2.imread(p), classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = True, classesConfidence=0.65))

    for d1 in data:
        paths = []
        for d2 in data:
            comp = ai.compareImages(imgData1=d1, imgData2=d2, compareWholeImages=True, minObjWeight=0.0)
            if comp > 0.001:
                paths.append((d2, comp))
        
        paths.sort(key=lambda x: x[1], reverse=True)
        cv2.imshow('org', cv2.resize(d1.orgImage, (512,512)))
        cv2.imshow('simmilar', np.concatenate([cv2.resize(path[0].orgImage, (256,256)) for path in paths[0:min(len(paths),8)]], axis=1))
        while cv2.waitKey(0) != ord(' '):
            time.sleep(0.5)
    return

if __name__ == "__main__":
    objectComparisonTest()
    print("Done")