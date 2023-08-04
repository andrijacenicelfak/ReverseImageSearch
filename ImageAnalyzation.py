from enum import Enum
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from ultralytics.nn.modules import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors
import json
from scipy.spatial.distance import cosine
import os

# Modify the functionality of the Detect class for intermediate layer extraction of features.
# The functionality remains the same; it just adds the final part.
def modifyDetectClass(): 
    Detect.callback = None

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'): 
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox =  dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        if self.export and self.format in ('tflite', 'edgetpu'):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        # Added code for intermediate data extraction
        if Detect.callback is not None:
            Detect.callback(x[1])
            Detect.callback = None
        #
        return y if self.export else (y, x)
    
    Detect.forward = forward
    return

class BoundingBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def toStr(self) -> str:
        return str(self.x1) + " : " + str(self.x2) + " :: " + str(self.y1) + " : "  + str(self.y2)


class ImageClassificationData:
    def __init__(self, className : str, boundingBox : BoundingBox, features : list = None):
        self.className : str =className
        self.boundingBox : BoundingBox = boundingBox
        self.features = features

class ImageData:
    def __init__(self, orgImage: None, classes : list[ImageClassificationData] = [], features : list = []):
        self.classes = classes
        self.features = features
        self.orgImage = orgImage
class AnalyzationType(Enum):
    FullVector = 0
    BMM = 1
    Avg = 2
    Cut = 3
class ImageAnalyzation:
    typeDict = {AnalyzationType.FullVector : None, AnalyzationType.BMM : "bmm", AnalyzationType.Avg : "avg", AnalyzationType.Cut : "2048"}
    def __init__(self, model : str, device : str = "cuda", analyzationType : AnalyzationType = AnalyzationType.Cut):
        self.model = YOLO(model)
        self.model.to(device)
        self.modelNames = self.model.names
        self.modelName = model
        self.vlist = None
        self.wholeVector = False
        if type != 0:
            if os.path.exists(model + "-" + self.typeDict[analyzationType] +  ".json"):
                with open(model + "-" + self.typeDict[analyzationType]  + ".json", "r") as   f:
                    self.vlist = json.load(fp=f)
            else:
                print("There is not vector for that model. You should first generate the model vector. Returninig the whole vector!")
                self.wholeVector = True
        else:
            self.wholeVector = True
        modifyDetectClass()
        # self.pca : PCA = joblib.load("yolov8s-pca.joblib")

    
    def getObjectClasses(self, image, objectFeatures = False, conf = 0.65) -> list[ImageClassificationData]:
        res = self.model.predict(image, verbose=False, conf=conf)
        data = [(self.modelNames[int(c.cls)], np.array(c.xyxy.cpu(), dtype="int").flatten()) for r in res for c in r.boxes]
        imcdata = [ImageClassificationData(className=d[0], boundingBox=BoundingBox(d[1][0], d[1][1], d[1][2], d[1][3]), features=None) for d in data]

        if objectFeatures:
            for d in range(len(imcdata)):
                imcdata[d].features = self.getFeatureVectorFromBounds(image, imcdata[d].boundingBox)
        return imcdata
    
    def getFeatureVector(self, image, wholeVector = False) -> list:
        self.value = None
        def setValue(x : any):
            self.value = x
        Detect.callback = setValue
        img2 = cv2.resize(image, dsize=(224,224), interpolation=cv2.INTER_LINEAR) #need to scale the image to be the same size, so the output vectors are the same size
        self.model.predict(img2, verbose=False)
        return self.reduceData(np.array(self.value.cpu()).flatten(), wholeVector)
    
    def getFeatureVectorFromBounds(self, image, boundigBox : BoundingBox = None, wholeVector = False) -> list:
        image = image[boundigBox.y1 : boundigBox.y2, boundigBox.x1 : boundigBox.x2]
        return self.getFeatureVector(image, wholeVector)
    
    def getImageData(self, image, classesData = True, imageFeatures = False, objectsFeatures = False, returnOriginalImage = False, wholeVector = False)-> ImageData:
        classes = None
        imgFeatures = None
        if classesData:
            classes = self.getObjectClasses(image, objectsFeatures)
        if imageFeatures:
            imgFeatures = self.getFeatureVector(image, wholeVector)
        return ImageData(classes= classes, features= imgFeatures, orgImage= image if returnOriginalImage else None)

    
    def compareImageHistograms(self, img1, img2):
        if img1 is None or img2 is None:
            raise Exception("Can't compare histograms if img1 and img2 are None")
        h1 = cv2.calcHist([img1], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.calcHist([img2], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        h2 = cv2.normalize(h2, h2).flatten()
        return 1 - cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    
    #compares the whole images with formula: similarity = objSimilarity * wholeImageSimilarity
    # objSimilarity = 2 * (sum(max(similarity(x,y)) - avg(nonmax(similarity(x,y))))) / (combinedNumObj + numOfDiffObj)
    # combinedNumObj is the combined numb of objects in two images
    # numOfDiffObj is number of classes that appear in on image and not the other and is they do appear number of different number of apperances
    # wholeImageSimilarity histogram comparison and descriptor comaparison
    def compareImages(self, imgData1 : ImageData, imgData2: ImageData, compareWholeImages = False, compareImageObjects = False, compareHistograms = True, containAllObjects = False):
        if imgData1.orgImage is None or imgData2.orgImage is None:
            raise Exception("Image data should contain the original image for comparesons")
        
        # if len(imgData1.classes) == 0 or len(imgData2.classes) == 0:
        #     compareImageObjects = False

        #compare whole images
        wholeImageFactor = 1
        if compareWholeImages:
            if imgData1.features is None:
                imgData1.features = self.getFeatureVector(imgData1.orgImage)

            if imgData2.features is None:
                imgData2.features = self.getFeatureVector(imgData2.orgImage)
            histWeight = 1
            if compareHistograms:
                histWeight = self.compareImageHistograms(imgData1.orgImage, imgData2.orgImage)
            wholeImageFactor = histWeight * (100 - cosine(imgData1.features, imgData2.features)*1000) / 100

        #Compare objects in images
        imageObjectsFactor = 1
        if compareImageObjects and ((containAllObjects and len(imgData1.classes) == len(imgData2.classes)) or not containAllObjects):
            numOfObjects = len(imgData1.classes) + len(imgData2.classes)
            classNames1 = list(map(lambda x : x.className, imgData1.classes))
            classNames2 = list(map(lambda x : x.className, imgData2.classes))

            diff1 = list(filter(lambda x : x not in classNames2 or classNames2.count(x) != classNames1.count(x), classNames1))
            diff2 = list(filter(lambda x : x not in classNames1 or classNames2.count(x) != classNames1.count(x), classNames2))

            numOfDifferentObjects = (len(diff1) + len(diff2))
            
            comparisonSum = 0
            for obj1 in imgData1.classes:
                comparisonMax = 0
                nonMaxSum =0 
                
                for obj2 in imgData2.classes:
                    curr = self.compareImageClassificationData(obj1, obj2, imgData1.orgImage, imgData2.orgImage, cutImage=True, compareHistograms=True)
                    comparisonMax = max(comparisonMax, curr)
                    nonMaxSum = nonMaxSum + curr
                nonMaxSum = nonMaxSum - comparisonMax
                comparisonSum = comparisonSum + comparisonMax - (nonMaxSum / max(len(imgData2.classes), 1)) #max - can have 0 objs, div with 0

            imageObjectsFactor = comparisonSum * 2 / max(1, numOfObjects + numOfDifferentObjects)

        return (wholeImageFactor * imageObjectsFactor)
    
    # compares two images by cutting the image and comparing two classes, returns "distance"
    def compareImageClassificationData(self, icd1 : ImageClassificationData, icd2 : ImageClassificationData, img1 = None, img2 = None, cutImage  = False, compareHistograms = True):
        if icd1.className != icd2.className:
            return 0
        
        #cut the image
        if cutImage:
            if img1 is None or img2 is None:
                raise Exception("Can't cut None type")
            img1 = img1[icd1.boundingBox.y1 : icd1.boundingBox.y2, icd1.boundingBox.x1 : icd1.boundingBox.x2]
            img2 = img2[icd2.boundingBox.y1 : icd2.boundingBox.y2, icd2.boundingBox.x1 : icd2.boundingBox.x2]

        #Compare histograms
        histWeight = 1
        if compareHistograms:
            histWeight = self.compareImageHistograms(img1, img2)

        #Get the features if they are missing
        if icd1.features is None:
            if img1 is None:
                raise Exception("There are no features for the image classification data and image is None for icd1")
            icd1.features = self.getFeatureVector(img1)
        if icd2.features is None:
            if img2 is None:
                raise Exception("There are no features for the image classification data and image is None for icd2")
            icd2.features = self.getFeatureVector(img2)

        dist = cosine(icd1.features, icd2.features) * 1000

        return (100 - dist)/100 * histWeight

    def reduceData(self, data, wholeVector = False):
        return data if wholeVector or self.wholeVector else data[self.vlist]
    
    def generateVector(self, imgPaths : list[str], minDist = 1.0e-8):
        diff = None
        diffNormalizedLast = None
        lastData = None
        for i in range(len(imgPaths)):
            data = self.getFeatureVector(cv2.imread(imgPaths[i]), wholeVector=True)
            if diff is None:
                print("data lenght : " , len(data))
                diff = np.zeros(len(data))
            if lastData is not None:
                diff = diff + np.absolute(lastData - data)
                diffNormalized = diff / (i+1)
                if diffNormalizedLast is not None:
                    dist = cosine(diffNormalizedLast, diffNormalized)
                    print(dist - minDist)
                    if minDist > dist:
                        break
                diffNormalizedLast = diffNormalized
            lastData = data
        #vector of 2048
        diffIndex = list(zip(diffNormalizedLast, range(len(diffNormalizedLast))))
        diffIndex.sort(reverse= True, key=lambda x: x[0])
        diffIndexCut = list(map(lambda x : x[1] ,diffIndex[:2048]))
        with open(self.modelName + "-2048.json", "w") as f:
            json.dump(diffIndexCut, fp=f)
        #vector from avg
        diffAvg = np.average(diffNormalizedLast)
        diffAvgIndexes = list(map(lambda y: y[1], filter(lambda x : x[0] > diffAvg ,zip(diffNormalizedLast, range(len(diffNormalizedLast))))))
        print(len(diffAvgIndexes))
        with open(self.modelName + "-avg.json", "w") as f:
            json.dump(diffAvgIndexes, fp=f)
        #vec between min max
        diffMin = np.min(diffNormalizedLast)
        diffMax = np.max(diffNormalizedLast)
        betweenMinMax = (diffMin + diffMax)/2
        diffBetweenMMIndexes = list(map(lambda y: y[1], filter(lambda x : x[0] > betweenMinMax ,zip(diffNormalizedLast, range(len(diffNormalizedLast))))))
        print(len(diffBetweenMMIndexes))
        with open(self.modelName + "-bmm.json", "w") as f:
            json.dump(diffBetweenMMIndexes, fp=f)
        return