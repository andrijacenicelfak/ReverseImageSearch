from ImageAnalyzationModule.ImageAnalyzationFile import *
from FileSystem.FileExplorerFile import FileExplorer
import time
import numpy as np
def objectComparisonTest():
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Documents\\dev\\ImageClassification\\YOLOv8\\imgs")
    paths = fe.search()

    ai = ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="1A-5")
    data: list[ImageData] = []
    for p in paths:
        data.append(ai.getImageData(cv2.imread(p), classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = True))

    for d in data:
        for cdata in d.classes:
            # if cdata.className.lower() != "dog":
            #     break
            img = d.orgImage[cdata.boundingBox.y1 : cdata.boundingBox.y2, cdata.boundingBox.x1 : cdata.boundingBox.x2]
            for md in data:
                if md == d:
                    continue
                for mcdata in md.classes:
                    if mcdata.className != cdata.className:
                        break
                    img2 = md.orgImage[mcdata.boundingBox.y1 : mcdata.boundingBox.y2, mcdata.boundingBox.x1 : mcdata.boundingBox.x2]
                    dist = ai.compareImageClassificationData(icd1=cdata, icd2=mcdata)
                    print(dist)
                    # cv2.putText(img2, "dist : " + str(dist), (5, 20), 1, 1, (255,0,0), 1, cv2.LINE_AA)
                    cv2.imshow("org", img)
                    cv2.imshow("f2", img2)
                    key = 1
                    while key != ord(' ') and key != 27:
                        key = cv2.waitKey(0)
                        time.sleep(0.4)
                    cv2.destroyAllWindows()
                    if key == 27:
                        break

def imageComparisonTest():
    fe = FileExplorer(startDirectory="C:\\Users\\best_intern\\Documents\\dev\\ImageClassification\\YOLOv8\\imgs")
    paths = fe.search()

    ai = ImageAnalyzation("yolov8s.pt", device="cuda", analyzationType=AnalyzationType.CoderDecoder, coderDecoderModel="1A-5")
    data: list[ImageData] = []
    for p in paths:
        data.append(ai.getImageData(cv2.imread(p), classesData = True, imageFeatures = True, objectsFeatures = True, returnOriginalImage = True))

    for d1 in data:
        paths = []
        for d2 in data:
            if d1 == d2:
                print("isto")
            comp = ai.compareImages(imgData1=d1, imgData2=d2, compareWholeImages=True)
            paths.append((d2.orgImage, comp))
        
        paths.sort(key=lambda x: x[1], reverse=True)
        cv2.imshow('org', cv2.resize(d1.orgImage, (512,512)))
        cv2.imshow('simmilar', np.concatenate([cv2.resize(path[0], (256,256)) for path in paths], axis=1))
        while cv2.waitKey(0) != ord(' '):
            time.sleep(0.5)
    return



if __name__ == "__main__":
    imageComparisonTest()
    print("Done")