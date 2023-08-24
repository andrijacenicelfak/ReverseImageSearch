class DisplayItem:
    def __init__(self,image_path,accuracy, image_data = None):
        self.image_path=image_path
        self.accuracy=accuracy
        self.image_data = image_data

class DisplayList:

    def __init__(self):
        self.items=[]

    def __iter__(self):
        return iter(self.items)
    
    def append(self,item):
        self.items.append(item)

    def filter_sort(self,val):
        self.items=[item for item in self.items if item.accuracy>=val]
        self.items.sort(key=lambda item: item.accuracy,reverse=True)       
 
    def clear(self):
        self.items.clear()

    def average(self):
        suma = 0
        if len(self.items) == 0:
            return 0
        maxel = max(self.items, key=lambda a: a.accuracy)
        for i in self.items:
            suma += i.accuracy
        suma -= maxel.accuracy
        return suma / max(1, len(self.items))