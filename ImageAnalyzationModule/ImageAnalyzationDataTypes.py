
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
