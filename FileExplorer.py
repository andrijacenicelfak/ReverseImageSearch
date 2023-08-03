import os

class FileExplorer:
    def __init__(self, startDirectory):
        self.startDirectory = startDirectory
        self.images = []
        self.image_extensions = ['.bmp', '.pbm', '.pgm', '.gif', '.sr', '.ras', '.jpeg', '.jpg', '.jpe', '.jp2', '.tiff', '.tif', '.png'] # list of image file formats supported by OpenCV
        self.loadImages()
        
    #ovo je za indexiranje treba da napravimo da radi sa bazom
    def search(self) -> list[str]:
        images = []
        for root, _, files in os.walk(self.startDirectory):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in self.image_extensions:
                    images.append(root + "\\" +file)
        self.images = images
        self.saveImages(images)
        return images
    #?? same shit koa i dole
    def getLastSearch(self) -> list[str]:
        return self.images
    #?? moze i da se zakomentarise za sad nije ni bitno
    def getChanges(self) -> list[str]:
        oldImages = self.images
        self.search()
        newImages = list(filter(lambda x : x not in oldImages, self.images))
        delImages = list(filter(lambda x : x not in newImages, oldImages))
        self.saveImages(newImages, append=True)
        return (newImages, delImages)
    #pisi u bazu
    def saveImages(self, images, append = False):
        #TODO database
        with open("images.txt",'a' if append else 'w') as f:
            for i in images:
                f.write("%s\n" % i)
        return
    #ucitaj iz baze
    def loadImages(self):
        #TODO database
        if not os.path.isfile("images.txt"):
            return
        with open("images.txt", 'r') as f:
            for i in f:
                self.images.append(i[:-1])
        return