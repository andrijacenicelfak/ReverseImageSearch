import os
import cv2
class FileExplorer:
    def __init__(self, startDirectory):
        self.startDirectory = startDirectory
        self.images = []
        self.image_extensions = ['.bmp', '.pbm', '.pgm', '.gif', '.sr', '.ras', '.jpeg', '.jpg', '.jpe', '.jp2', '.tiff', '.tif', '.png'] # list of image file formats supported by OpenCV
        # self.loadImages()
        
    #ovo je za indexiranje treba da napravimo da radi sa bazom
    # def search(self) -> list[str]:
    #     images = []
    #     for root, _, files in os.walk(self.startDirectory):
    #         for file in files:
    #             _, ext = os.path.splitext(file)
    #             if ext in self.image_extensions:
    #                 images.append(root + "\\" +file)
    #     self.images = images
    #     self.saveImages(images)
    #     return images
    
    def search2(self):
        file_list = os.listdir(self.startDirectory)
        yield_this=[]
        counter=0
    
        for file_name in file_list:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.startDirectory = os.path.normpath(self.startDirectory)
                
                image_path = os.path.join(self.startDirectory , file_name)
                
                image = cv2.imread(image_path)       
                if image is not None:
                    
                    yield_this.append((image_path,image))
                    counter+=1
                else:
                    
                    print(f"Unable to read image: {file_name}")
                if counter==100:
                    
                    counter=0
                    yield yield_this
                    yield_this.clear()

        yield yield_this
        
    #?? same shit koa i dole
    # def getLastSearch(self) -> list[str]:
    #     return self.images
    # #?? moze i da se zakomentarise za sad nije ni bitno
    # def getChanges(self) -> list[str]:
    #     oldImages = self.images
    #     self.search()
    #     newImages = list(filter(lambda x : x not in oldImages, self.images))
    #     delImages = list(filter(lambda x : x not in newImages, oldImages))
    #     self.saveImages(newImages, append=True)
    #     return (newImages, delImages)
    # #pisi u bazu
    # def saveImages(self, images, append = False):
    #     #TODO database
    #     with open("images.txt",'a' if append else 'w') as f:
    #         for i in images:
    #             f.write("%s\n" % i)
    #     return
    # #ucitaj iz baze
    # def loadImages(self):
    #     #TODO database
    #     if not os.path.isfile("images.txt"):
    #         return
    #     with open("images.txt", 'r') as f:
    #         for i in f:
    #             self.images.append(i[:-1])
    #     return
    