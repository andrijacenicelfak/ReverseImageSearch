class SearchParams:
    def __init__(self, *, compareObjects = True, 
                         compareWholeImages = True, 
                         maxWeightReduction = True,
                         containSameObjects = False,
                         confidenceCalculation = False,
                         magnitudeCalculation = False,
                         textContext = False,
                         minObjConf = 0.5,
                         minObjWeight = 0.05,
                         textContextWeight = 0.6,
                         selectedIndex = None,
                         imagePath = ".\\AppImages\\noimg.png"
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
        self.textContext = textContext
        self.textContextWeight = textContextWeight
        self.data = None
    
    def from_dict(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])
    
    def get_dict(self):
        d = dict()
        d["compareObjects"] = self.compareObjects
        d["compareWholeImages"] = self.compareWholeImages
        d["maxWeightReduction"] = self.maxWeightReduction
        d["containSameObjects"] = self.containSameObjects
        d["magnitudeCalculation"] = self.magnitudeCalculation
        d["magnitudeCalculation"] = self.magnitudeCalculation
        d["textContext"] = self.textContext
        d["textContextWeight"] = self.textContextWeight
        d["minObjConf"] = self.minObjConf
        d["minObjWeight"] = self.minObjWeight
        return d