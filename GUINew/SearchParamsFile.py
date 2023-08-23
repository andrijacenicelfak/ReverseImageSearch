class SearchParams:
    def __init__(self, *, compareObjects = True, 
                         compareWholeImages = True, 
                         maxWeightReduction = True,
                         containSameObjects = False,
                         confidenceCalculation = False,
                         magnitudeCalculation = False,
                         minObjConf = 0.5,
                         minObjWeight = 0.05,
                         selectedIndex = None,
                         imagePath = ".\\AppImages\\noimg.jpeg"
                         ):
        self.compareObjects = compareObjects
        self.compareWholeImages = compareWholeImages
        self.maxWeightReduction = maxWeightReduction
        self.containSameObjects = containSameObjects
        self.confidenceCalculation = confidenceCalculation
        self.magnitudeCalculation = magnitudeCalculation
        self.minObjConf = minObjConf
        self.minObjWeight = minObjWeight
        self.selectedIndex = selectedIndex
        self.imagePath = imagePath
        self.data = None