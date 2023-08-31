from enum import Enum
from functools import reduce
from math import sqrt
import time
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from ultralytics.nn.modules import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors
import json
from scipy.spatial.distance import cosine
import os
import torchvision
from DB.Functions import get_image_flag
from ImageAnalyzationModule.ConvolutionalModels import AutoEncoderDecoder

MAX_SIMMILARITY = 100
IMAGE_SIZE_AUTOENCODER = (128, 128)
IMAGE_SIZE_AUTOENCODER_1D = 128
IMAGE_SIZE_YOLO = (224, 224)
RUN_THOUGH_SIZE = (64, 64, 3)


class AnalyzationType(Enum):
    FullVector = 0
    BMM = 1
    Avg = 2
    Cut = 3
    CoderDecoder = 4


typeDict = {
    AnalyzationType.FullVector: None,
    AnalyzationType.BMM: "bmm",
    AnalyzationType.Avg: "avg",
    AnalyzationType.Cut: "2048",
    AnalyzationType.CoderDecoder: None,
}


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
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in (
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
        ):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = (
            dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)
            * self.strides
        )
        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor(
                [img_w, img_h, img_w, img_h], device=dbox.device
            ).reshape(1, 4, 1)
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
        return (
            str(self.x1)
            + " : "
            + str(self.x2)
            + " :: "
            + str(self.y1)
            + " : "
            + str(self.y2)
        )

    def __eq__(self, other) -> bool:
        return (
            self.x1 == other.x1
            and self.x2 == other.x2
            and self.y1 == other.y1
            and self.y2 == other.y2
        )


class ImageClassificationData:
    def __init__(
        self,
        className: str,
        boundingBox: BoundingBox,
        features: list = None,
        weight: float = 0,
        conf: float = 0,
    ):
        self.className: str = className
        self.boundingBox: BoundingBox = boundingBox
        self.features = features
        self.weight: float = weight
        self.conf = conf
        self.id = None

    def __eq__(self, other) -> bool:
        return (
            self.className == other.className
            and self.boundingBox == other.boundingBox
            and self.features == other.features
            and self.weight == other.weight
        )


class ImageData:
    def __init__(
        self,
        orgImage: None,
        classes: list[ImageClassificationData] = [],
        features: list = [],
        histogram=None,
    ):
        self.classes = classes
        self.features = features
        self.orgImage = orgImage
        self.histogram = histogram

    # def __eq__(self, other) -> bool:
    #     return self.classes == other.classes and self.features == other.features and self.orgImage == other.orgImage


class ImageAnalyzation:
    def __init__(
        self,
        model: str,
        *,
        device: str = "cuda",
        analyzationType: AnalyzationType = AnalyzationType.CoderDecoder,
        aedType: type = AutoEncoderDecoder,
        coderDecoderModel: str = None,
        normalization = True
    ):
        model = model.split(".")[0]
        self.model = YOLO(model)
        self.model.to(device)
        self.device = device
        self.modelNames = self.model.names
        self.modelName = model
        self.vlist = None
        self.wholeVector = True
        self.coderDecoder = False

        if analyzationType == AnalyzationType.CoderDecoder:
            self.t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5), (0.5))
                ]
            )
            self.coderDecoderModel = aedType(normalization = normalization)
            self.coderDecoderModel.eval()
            self.coderDecoderModel.load_state_dict(
                torch.load(f".\\models\\{coderDecoderModel}.model")
            )
            self.coderDecoderModel.to(device, non_blocking=True)
            self.coderDecoder = True
        elif analyzationType != AnalyzationType.FullVector:
            vectorPath = f".\\models\\{model}-{typeDict[analyzationType]}.json"
            if os.path.exists(vectorPath):
                with open(vectorPath, "r") as f:
                    self.vlist = json.load(fp=f)
                self.wholeVector = False
                modifyDetectClass()
            else:
                print(
                    "There is no vector for that model. You should first generate the model vector. Returninig the whole vector!"
                )
        self.runThrough()

    # To load the model on GPU run a small image through
    def runThrough(self):
        self.model.predict(np.zeros(RUN_THOUGH_SIZE), verbose=False)

    # Calls the yolo model to get the bounding boxes and classes on the image
    def getObjectClasses(
        self, image, *, objectFeatures=False, conf=0.55
    ) -> list[ImageClassificationData]:
        res = self.model.predict(image, verbose=False, conf=conf)
        data = [
            (
                self.modelNames[int(c.cls)],
                np.array(c.xyxy.cpu(), dtype="int").flatten(),
                c.conf.cpu().detach().item(),
            )
            for r in res
            for c in r.boxes
        ]
        imcdata = []
        for d in data:
            bbox = BoundingBox(d[1][0], d[1][1], d[1][2], d[1][3])
            weight = ((bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)) / (
                image.shape[0] * image.shape[1]
            )
            imcdata.append(
                ImageClassificationData(
                    className=d[0],
                    boundingBox=bbox,
                    features=None,
                    weight=weight,
                    conf=d[2],
                )
            )

        if objectFeatures:
            for d in range(len(imcdata)):
                imcdata[d].features = self.getFeatureVectorFromBounds(
                    image, imcdata[d].boundingBox
                )
        return imcdata

    # Extracts feature vector from yolo model
    def getFeatureVector(self, image, *, wholeVector=False) -> list:
        self.value = None

        def setValue(x: any):
            self.value = x

        Detect.callback = setValue
        img2 = cv2.resize(
            image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR
        )  # need to scale the image to be the same size, so the output vectors are the same size
        self.model.predict(img2, verbose=False)
        return self.reduceData(np.array(self.value.cpu()).flatten(), wholeVector)

    # Extracts feature vector from the object(bounding box), can extract from yolo and from AutoEncoderDecoder
    def getFeatureVectorFromBounds(
        self, image, boundigBox: BoundingBox = None, wholeVector=False
    ) -> list:
        image = image[boundigBox.y1 : boundigBox.y2, boundigBox.x1 : boundigBox.x2]
        if not self.coderDecoder:
            return self.getFeatureVector(image, wholeVector=wholeVector)
        else:
            img = cv2.resize(image, IMAGE_SIZE_AUTOENCODER)  # .transpose(2, 1, 0)
            img = self.t(img)
            return self.getCodedFeatureVector(
                images=torch.stack(
                    [
                        img,
                    ]
                )
            ).flatten()

    # returns ImageData with the selected params
    def getImageData(
        self,
        image,
        *,
        classesData=True,
        imageFeatures=False,
        objectsFeatures=False,
        returnOriginalImage=False,
        classesConfidence=0.65,
        wholeVector=False,
    ) -> ImageData:
        classes = None
        imgFeatures = None
        if classesData:
            # ctime = time.time()
            classes = self.getObjectClasses(
                image, objectFeatures=objectsFeatures, conf=classesConfidence
            )
            # print(f"CTime : {time.time()-ctime}")
        if imageFeatures:
            # ftime = time.time()
            if self.coderDecoder:
                imgFeatures = self.getFetureVectorAutocoder(image)
            else:
                imgFeatures = self.getFeatureVector(image, wholeVector=wholeVector)
            # print(f"FVTime : {time.time()-ftime}")

        imgData = ImageData(
            classes=classes,
            features=imgFeatures,
            orgImage=image if returnOriginalImage else None,
        )
        # histogram=self.generateHistogram(imageData=imgData, img=image)
        # imgData.histogram = histogram
        return imgData

    def generateHistogram(self, imageData: ImageData, img):
        mask = np.full(img.shape[:2], fill_value=255, dtype="uint8")
        for c in imageData.classes:
            bbox = c.boundingBox
            cv2.rectangle(mask, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 0, 0), -20)
        # cv2.imshow("hist", cv2.resize(mask, (256, 256)))
        # cv2.imshow("img", cv2.resize(img, (256,256)))
        h = cv2.calcHist([img], [0, 1, 2], mask, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        cv2.normalize(h, h)
        return h

    def compareHistograms(self, h1, h2):
        return 1 - cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    def compareImageHistograms(self, img1, img2):
        if img1 is None or img2 is None:
            raise Exception("Can't compare histograms if img1 and img2 are None")
        h1 = cv2.calcHist(
            [img1], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256]
        )
        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.calcHist(
            [img2], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256]
        )
        h2 = cv2.normalize(h2, h2).flatten()
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) + 1e-4

    # compares the whole images with formula: similarity = objSimilarity * wholeImageSimilarity
    # objSimilarity = 2 * (sum(max(similarity(x,y)) - avg(nonmax(similarity(x,y))))) / (combinedNumObj + numOfDiffObj)
    # combinedNumObj is the combined numb of objects in two images
    # numOfDiffObj is number of classes that appear in on image and not the other and is they do appear number of different number of apperances
    # wholeImageSimilarity histogram comparison and descriptor comaparison
    def compareImagesV1(
        self,
        imgData1: ImageData,
        imgData2: ImageData,
        *,
        compareWholeImages=False,
        compareImageObjects=False,
        compareHistograms=True,
        containAllObjects=False,
    ):
        print("[DEPRICATED] Using depricated code : comapareImagesOld")
        if imgData1.orgImage is None or imgData2.orgImage is None:
            raise Exception(
                "Image data should contain the original image for comparesons"
            )

        # if len(imgData1.classes) == 0 or len(imgData2.classes) == 0:
        #     compareImageObjects = False

        # compare whole images
        wholeImageFactor = 1
        if compareWholeImages:
            if imgData1.features is None:
                imgData1.features = self.getFeatureVector(imgData1.orgImage)

            if imgData2.features is None:
                imgData2.features = self.getFeatureVector(imgData2.orgImage)
            histWeight = 1
            if compareHistograms:
                histWeight = self.compareImageHistograms(
                    imgData1.orgImage, imgData2.orgImage
                )
            wholeImageFactor = (
                histWeight
                * (
                    100
                    - cosine(imgData1.features.flatten(), imgData2.features.flatten())
                    * 1000
                )
                / 100
            )

        # Compare objects in images
        imageObjectsFactor = 1
        if compareImageObjects and (
            (containAllObjects and len(imgData1.classes) == len(imgData2.classes))
            or not containAllObjects
        ):
            numOfObjects = len(imgData1.classes) + len(imgData2.classes)
            classNames1 = list(map(lambda x: x.className, imgData1.classes))
            classNames2 = list(map(lambda x: x.className, imgData2.classes))

            diff1 = list(
                filter(
                    lambda x: x not in classNames2
                    or classNames2.count(x) != classNames1.count(x),
                    classNames1,
                )
            )
            diff2 = list(
                filter(
                    lambda x: x not in classNames1
                    or classNames2.count(x) != classNames1.count(x),
                    classNames2,
                )
            )

            numOfDifferentObjects = len(diff1) + len(diff2)

            comparisonSum = 0
            for obj1 in imgData1.classes:
                comparisonMax = 0
                nonMaxSum = 0

                for obj2 in imgData2.classes:
                    curr = self.compareImageClassificationData(
                        obj1,
                        obj2,
                        img1=imgData1.orgImage,
                        img2=imgData2.orgImage,
                        cutImage=True,
                        compareHistograms=True,
                    )
                    comparisonMax = max(comparisonMax, curr)
                    nonMaxSum = nonMaxSum + curr
                nonMaxSum = nonMaxSum - comparisonMax
                comparisonSum = (
                    comparisonSum
                    + comparisonMax
                    - (nonMaxSum / max(len(imgData2.classes), 1))
                )  # max - can have 0 objs, div with 0

            imageObjectsFactor = (
                comparisonSum * 2 / max(1, numOfObjects + numOfDifferentObjects)
            )

        return wholeImageFactor * imageObjectsFactor

    def compareImages(
        self,
        *,
        imgData1: ImageData,
        imgData2: ImageData,
        compareObjects=True,
        compareWholeImages=True,
        maxWeightReduction=True,
        containSameObjects=False,
        confidenceCalculation=False,
        magnitudeCalculation=False,
        minObjConf=0.5,
        minObjWeight=0.05,
        selectedIndex=None,
    ):
        # comparing whole images, if match return

        imageComparison = 1
        if compareWholeImages:
            imageComparison = 1 - cosine(imgData1.features, imgData2.features)
            if imageComparison == 1:
                return MAX_SIMMILARITY

        # if len(imgData1.classes) == 0 and len(imgData2.classes) == 0:
        #     compareObjects = False

        if selectedIndex is not None and selectedIndex < len(imgData1.classes):
            selectedClassdata = imgData1.classes[selectedIndex]
            imgData1.classes = [
                selectedClassdata,
            ]
        else:
            imgData1.classes = list(
                filter(
                    lambda x: x.weight >= minObjWeight and x.conf >= minObjConf,
                    imgData1.classes,
                )
            )
            imgData2.classes = list(
                filter(
                    lambda x: x.weight >= minObjWeight and x.conf >= minObjConf,
                    imgData2.classes,
                )
            )

        flag1 = get_image_flag(list(map(lambda x: x.className, imgData1.classes)))
        flag1 = (flag1[0], flag1[1], flag1[0].bit_count() + flag1[1].bit_count())
        flag2 = get_image_flag(list(map(lambda x: x.className, imgData2.classes)))
        flag2 = (flag2[0], flag2[1], flag2[0].bit_count() + flag2[1].bit_count())

        flag = (flag1[0] & flag2[0], flag1[1] & flag2[1])
        flag = (flag[0], flag[1], flag[0].bit_count() + flag[1].bit_count())

        # if the objects from the first image are not matched then image is not a match
        if containSameObjects and flag[2] != flag1[2]:
            return 0

        if flag[2] == 0:
            return 0

        objectComparison = 1
        if compareObjects:
            objectComparison = self.objectComparison(
                imgData1=imgData1,
                imgData2=imgData2,
                selectedIndex=selectedIndex,
                maxWeightReduction=maxWeightReduction,
                confidenceCalculation=confidenceCalculation,
                magnitudeCalculation=magnitudeCalculation,
            )

        return objectComparison * imageComparison

    # function for object comparison
    def objectComparison(
        self,
        *,
        imgData1: ImageData,
        imgData2: ImageData,
        selectedIndex=None,
        maxWeightReduction=False,
        confidenceCalculation=False,
        magnitudeCalculation=False,
    ):
        objectComparison = 1
        if len(imgData1.classes) == 0:
            return 1
        fts, stf = self.generateDicts(
            imgData1=imgData1,
            imgData2=imgData2,
            confidenceCalculation=confidenceCalculation,
            magnitudeCalculation=magnitudeCalculation,
        )
        if selectedIndex is not None:
            return fts[0][1]
        sumAll = 0
        numOfMatches = 0
        for i in range(len(imgData1.classes)):
            j = fts[i][0]
            if j != -1 and stf[j][0] == i:
                sumAll += (fts[i][1] / imgData1.classes[i].weight) * imgData2.classes[j].weight
                numOfMatches += 1

        objectComparison = (
            sumAll
            * 2
            / (max(1, len(imgData1.classes) + len(imgData2.classes)))
            * (numOfMatches / max(len(imgData1.classes), 1))
        )
        # if the object type with max weight (sum of weights of the same type) does not appear in the seccond image the match is reduced
        if maxWeightReduction:
            weights1 = dict()
            for obj in imgData1.classes:
                if obj.className in weights1:
                    weights1[obj.className] += obj.weight * obj.conf
                else:
                    weights1[obj.className] = obj.weight * obj.conf
            weights2 = dict()
            for obj in imgData2.classes:
                if obj.className in weights2:
                    weights2[obj.className] += obj.weight * obj.conf
                else:
                    weights2[obj.className] = obj.weight * obj.conf
            maxWeightObj = max(
                [key for key in weights1.keys()], key=lambda a: weights1[a]
            )
            if maxWeightObj not in map(lambda a: a.className, imgData2.classes):
                objectComparison *= 1e-6
        return objectComparison

    # compares the images by calculating the max object matching and similarity between images
    # objectComparison = the sum of all the max matches between objects, if two objects are most simmilar
    # with each other their simmilarity goes in the sum, the objects that don't match don't
    # if there is a match in comparing whole images the func returns 100
    def compareImagesV2(
        self,
        *,
        imgData1: ImageData,
        imgData2: ImageData,
        compareObjects=True,
        compareWholeImages=False,
        minObjWeight=0.05,
    ):
        # comparing whole images, if match return
        imageComparison = 1
        if compareWholeImages:
            imageComparison = 1 - cosine(imgData1.features, imgData2.features)
            if imageComparison == 1:
                return 100

        # check if the classes match, need to have simmilar classes
        imgData1.classes = list(
            filter(lambda x: x.weight > minObjWeight, imgData1.classes)
        )
        imgData2.classes = list(
            filter(lambda x: x.weight > minObjWeight, imgData2.classes)
        )

        numberOfObjects = 1e-4

        objects1 = list(map(lambda x: x.className, imgData1.classes))
        objects2 = list(map(lambda x: x.className, imgData2.classes))

        for o in objects1:
            if o in objects2:
                numberOfObjects += 1

        # If the class with the most weight does not exist in the second image lower the simmilarity
        maxWeightComponent = 1
        maxWeightClass = max(imgData1.classes, key=lambda x: x.weight)
        if maxWeightClass.className not in objects2:
            maxWeightComponent = 1e-2

        if len(imgData1.classes) == 0 and len(imgData2.classes) == 0:
            compareObjects = False

        # compare the objects within the image
        objectComparison = 1
        if compareObjects:
            fts, stf = self.generateDicts(imgData1=imgData1, imgData2=imgData2)
            sumAll = 0
            numOfMatches = 0
            for i in range(len(imgData1.classes)):
                j = fts[i][0]
                if j != -1 and stf[j][0] == i:
                    sumAll += (
                        fts[i][1] / imgData1.classes[i].weight
                    ) * imgData2.classes[j].weight
                    numOfMatches += 1
            objectComparison = (
                sumAll
                * 2
                / (max(1, len(imgData1.classes) + len(imgData2.classes)))
                * (numOfMatches / max(len(imgData1.classes), 1))
            )

        return (
            objectComparison
            * imageComparison
            * ((numberOfObjects / max(1, len(objects1))) ** 2)
            * maxWeightComponent
        )

    # Generates the dictionaries for comparing classification data
    # fts (first to second) generates the indexes of most simmilar objects for the list of objects from the first imgData
    # stf (second to first) generates the indexes of most simmilar object for the list of objects from the seconda imgData
    def generateDicts(
        self,
        *,
        imgData1: ImageData,
        imgData2: ImageData,
        confidenceCalculation=False,
        magnitudeCalculation=False,
    ) -> (dict, dict):
        fts = dict()
        stf = dict()
        for i in range(len(imgData1.classes)):
            fts[i] = (-1, 0)
        for i in range(len(imgData2.classes)):
            stf[i] = (-1, 0)

        for i, classData1 in enumerate(imgData1.classes):
            for j, classData2 in enumerate(imgData2.classes):
                sim = self.compareImageClassificationData(
                    icd1=classData1,
                    icd2=classData2,
                    confidenceCalculation=confidenceCalculation,
                    magnitudeCalculation=magnitudeCalculation,
                )
                if fts[i][1] <= sim:
                    fts[i] = (j, sim)
                if stf[j][1] <= sim:
                    stf[j] = (i, sim)

        return (fts, stf)

    # Calculates the similarity of objects on image
    def compareImageClassificationData(
        self,
        *,
        icd1: ImageClassificationData,
        icd2: ImageClassificationData,
        treshhold=0.1,
        confidenceCalculation=False,
        magnitudeCalculation=False,
        classNameComparison=True,
        scaleDown=True,
        scale=(0.9, 10),
    ):
        if icd1.features is None or icd2.features is None:
            raise Exception("Feature vector is None!")
        if icd1.weight is None or icd2.weight is None:
            raise Exception("No weights for calculating similarity!")
        if classNameComparison and icd1.className != icd2.className:
            return 0

        m1 = 0
        m2 = 0
        dist = 0
        v1 = []
        v2 = []
        for i in range(len(icd1.features)):
            f1, f2 = icd1.features[i], icd2.features[i]
            if f1 != f2:
                v1.append(f1)
                v2.append(f2)
                m1 += f1*f1
                m2 += f2*f2
                dist += f1 * f2
        m1 = sqrt(m1)
        m2 = sqrt(m2)
        dist = 1 - sqrt(dist) / max(m1 + m2, 1)
        # dist = 1 - cosine(v1, v2)

        with open('test.txt', 'w') as f:
            for f1, f2 in (zip(v1, v2)):
                f.write(f"{f1}\t{f2}\n")


        if magnitudeCalculation and (m1 != 0 and m2 !=0):  # Better results
            dist *= min(m1, m2) / max(m1, m2)

        # The values seem to be between example 0.6 - 1.0
        if scaleDown:
            # Values 0.0 - 1.0
            dist = (dist - scale[0]) * scale[1]

        if confidenceCalculation:  # Worse results
            dist = dist * icd1.conf * icd2.conf
        return min(dist, 1.0) if dist >= treshhold else 0

    # compares two images by cutting the image and comparing two classes, returns "distance"
    def compareImageClassificationDataOld(
        self,
        icd1: ImageClassificationData,
        icd2: ImageClassificationData,
        *,
        img1=None,
        img2=None,
        cutImage=False,
        compareHistograms=True,
    ):
        print("[DEPRICATED] Using depricated code : compareImageClassificationDataOld")

        if icd1.className != icd2.className:
            return 0

        # cut the image
        if cutImage:
            if img1 is None or img2 is None:
                raise Exception("Can't cut None type")
            img1 = img1[
                icd1.boundingBox.y1 : icd1.boundingBox.y2,
                icd1.boundingBox.x1 : icd1.boundingBox.x2,
            ]
            img2 = img2[
                icd2.boundingBox.y1 : icd2.boundingBox.y2,
                icd2.boundingBox.x1 : icd2.boundingBox.x2,
            ]

        # Compare histograms
        histWeight = 1
        if compareHistograms:
            histWeight = self.compareImageHistograms(img1, img2)

        # Get the features if they are missing
        if icd1.features is None:
            if img1 is None:
                raise Exception(
                    "There are no features for the image classification data and image is None for icd1"
                )
            icd1.features = self.getFeatureVector(img1)
        if icd2.features is None:
            if img2 is None:
                raise Exception(
                    "There are no features for the image classification data and image is None for icd2"
                )
            icd2.features = self.getFeatureVector(img2)

        dist = cosine(icd1.features, icd2.features) * 1000

        return (100 - dist) / 100 * histWeight

    # Reduce the vector size
    def reduceData(self, data, wholeVector=False):
        return data if wholeVector or self.wholeVector else data[self.vlist]

    # generates the vector for all the images
    def generateVector(self, imgPaths: list[str], minDist=1.0e-8):
        diff = None
        diffNormalizedLast = None
        lastData = None
        i = 0
        dist = 100
        while i < len(imgPaths) and minDist <= dist:
            data = self.getFeatureVector(cv2.imread(imgPaths[i]), wholeVector=True)
            if diff is None:
                print("data lenght : ", len(data))
                diff = np.zeros(len(data))
            if lastData is not None:
                diff = diff + np.absolute(lastData - data)
                diffNormalized = diff / (i + 1)
                if diffNormalizedLast is not None:
                    dist = cosine(diffNormalizedLast, diffNormalized)
                    print(dist - minDist)
                diffNormalizedLast = diffNormalized
            i += 1
            lastData = data
        # vector of 2048
        diffIndex = list(zip(diffNormalizedLast, range(len(diffNormalizedLast))))
        diffIndex.sort(reverse=True, key=lambda x: x[0])
        diffIndexCut = list(map(lambda x: x[1], diffIndex[:2048]))
        with open(self.modelName + "-2048.json", "w") as f:
            json.dump(diffIndexCut, fp=f)
        # vector from avg
        diffAvg = np.average(diffNormalizedLast)
        diffAvgIndexes = list(
            map(
                lambda y: y[1],
                filter(
                    lambda x: x[0] > diffAvg,
                    zip(diffNormalizedLast, range(len(diffNormalizedLast))),
                ),
            )
        )
        print(len(diffAvgIndexes))
        with open(self.modelName + "-avg.json", "w") as f:
            json.dump(diffAvgIndexes, fp=f)
        # vec between min max
        diffMin = np.min(diffNormalizedLast)
        diffMax = np.max(diffNormalizedLast)
        betweenMinMax = (diffMin + diffMax) / 2
        diffBetweenMMIndexes = list(
            map(
                lambda y: y[1],
                filter(
                    lambda x: x[0] > betweenMinMax,
                    zip(diffNormalizedLast, range(len(diffNormalizedLast))),
                ),
            )
        )
        print(len(diffBetweenMMIndexes))
        with open(self.modelName + "-bmm.json", "w") as f:
            json.dump(diffBetweenMMIndexes, fp=f)
        return

    # gets the feature vector extracted from the encoder of autoencoderdecoder model
    def getCodedFeatureVector(self, images):
        with torch.no_grad():
            return (
                self.coderDecoderModel.encode(images.to(self.device, non_blocking=True))
                .view((-1, self.coderDecoderModel.vectorLenght))
                .cpu()
                .detach()
                .numpy()
            )

    def getImageDataList(
        self,
        images,
        *,
        classesData=True,
        imageFeatures=False,
        objectsFeatures=False,
        returnOriginalImage=False,
        wholeVector=False,
    ) -> list[ImageData]:
        classes = []
        imgFeatures = []
        if classesData:
            classes = self.getObjectClassesList(images, objectFeatures=objectsFeatures)
        if imageFeatures:
            print(
                "Extracting image features with autoencoder NOT with YOLO, yolo's features for whole image comaprison is better than autoencoder"
            )
            # Not the same as getImageData, can't extract multiple image vectors from yolo only 1 at a time, this uses autoencoder.
            # Better results with yolo features
            imgFeatures = self.getFeatureVectorList(
                torch.stack(
                    [self.t(cv2.resize(img, IMAGE_SIZE_AUTOENCODER)) for img in images],
                    dim=0,
                ),
                wholeVector=wholeVector,
            )
        return [
            ImageData(
                orgImage=(images[i] if returnOriginalImage else None),
                classes=classes[i] if classesData else [],
                features=imgFeatures[i] if imageFeatures else [],
            )
            for i in range(len(images))
        ]

    # Gets the object classes in a list of images, better use getObjectClasses
    def getObjectClassesList(
        self, images, *, objectFeatures=True, conf=0.65
    ) -> list[list[ImageClassificationData]]:
        reses = self.model.predict(images, stream=True, verbose=False, conf=conf)
        data = []
        for res in reses:
            d = [
                (
                    self.modelNames[int(c.cls)],
                    np.array(c.xyxy.cpu(), dtype="int").flatten(),
                )
                for r in res
                for c in r.boxes
            ]
            data.append(d)
        imcdata = []
        imagescut: list[list[ImageClassificationData]] = []
        index = 0
        for i in range(len(data)):
            imgData: list[ImageClassificationData] = []
            for j in range(len(data[i])):
                d = data[i][j]
                bbox = BoundingBox(d[1][0], d[1][1], d[1][2], d[1][3])
                img = images[i][bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
                if self.coderDecoder:
                    img = self.t(cv2.resize(img, IMAGE_SIZE_AUTOENCODER))
                else:
                    img = cv2.resize(img, IMAGE_SIZE_YOLO)
                imagescut.append(img)
                weight = ((bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)) / (
                    images[i].shape[0] * images[i].shape[1]
                )
                classData = ImageClassificationData(
                    className=d[0], boundingBox=bbox, features=None, weight=weight
                )
                classData.id = index
                index += 1
                imgData.append(classData)
            imcdata.append(imgData)

        if objectFeatures:
            index2 = 0
            vectorData = self.getFeatureVectorList(images=torch.stack(imagescut, dim=0))
            for i in range(len(imcdata)):
                for j in range(len(imcdata[i])):
                    if imcdata[i][j].id == index2:
                        imcdata[i][j].features = vectorData[index2]
                        index2 += 1
                    else:
                        raise Exception("Error in seting features!")

        return imcdata

    def getFeatureVectorList(self, images, wholeVector=False) -> list:
        if not self.coderDecoder:
            print(
                "Getting the feature vector list from yolov8 is imposible, it has to be done 1 by one image, try using AnalyzationType.CoderDecoder"
            )
            ret = []
            for img in images:
                ret.append(self.getFeatureVector(img, wholeVector=wholeVector))
            return ret
        else:
            return self.getCodedFeatureVector(images=images)

    def getFetureVectorAutocoder(self, image):
        images = []
        for i in range(0, 2):
            size = (
                IMAGE_SIZE_AUTOENCODER_1D * (i + 1),
                IMAGE_SIZE_AUTOENCODER_1D * (i + 1),
            )
            img = cv2.resize(image, size)
            for x in range(int(size[0] / IMAGE_SIZE_AUTOENCODER_1D)):
                for y in range(int(size[1] / IMAGE_SIZE_AUTOENCODER_1D)):
                    images.append(
                        img[
                            y
                            * IMAGE_SIZE_AUTOENCODER_1D : (y + 1)
                            * IMAGE_SIZE_AUTOENCODER_1D,
                            x
                            * IMAGE_SIZE_AUTOENCODER_1D : (x + 1)
                            * IMAGE_SIZE_AUTOENCODER_1D,
                        ]
                    )
        imagest = list(map(lambda x: self.t(x), images))
        return (
            self.coderDecoderModel.encode(
                torch.stack(imagest).to(self.device, non_blocking=True)
            )
            .cpu()
            .detach()
            .numpy()
            .flatten()
        )
