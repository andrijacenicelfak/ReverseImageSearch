import sqlite3
import pickle
import time
from ImageAnalyzation import ImageAnalyzation
model_names = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}

def get_image_flag(terms):
    flag0 = 0
    flag1 = 0
    for term in terms:
        i = model_names[term]
        if i < 40:
            x = 2**i
            flag0 = flag0 | x
        else:
            i = i % 40
            x = 2**i            
            flag1 = flag1 | x
    # return flag.to_bytes((flag.bit_length()+7)//8,byteorder='big')
    return (flag0, flag1)

def decode_image_flag(flag):
    decoded_terms = []
    for term, bit_position in model_names.items():
        if flag & (2 ** bit_position):
            decoded_terms.append(term)
    return decoded_terms
  
class ImageDB:
    def __init__(self):
        print()
    def open_connection(self):
        self.con = sqlite3.connect("test3.db")
        self.cursor = self.con.cursor()
        try:
            self.con = sqlite3.connect("test3.db")
            self.cursor = self.con.cursor()
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    id INTEGER PRIMARY KEY,
                    image_id INT,
                    class_name TEXT NOT NULL,
                    desc BLOB NOT NULL,
                    weight REAL NOT NULL,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                );
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS images(
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL,
                    flag0 INTEGER NOT NULL,
                    flag1 INTEGER NOT NULL,
                    desc BLOB NOT NULL         
                );
            """)        
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_objects_image_id ON objects(image_id);")    
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_flag ON images(flag0,flag1);")       
   
            self.con.commit()
        except sqlite3.Error as e:
            print(f'Error occurred: {e}')

        
    def addImage(self,dbstruct:ImageAnalyzation.ImageData):#orgImage is image but here is path lol
        try:
            flag0, flag1 = get_image_flag([x.className for x in dbstruct.classes])
            self.cursor.execute('INSERT INTO images (path, flag0, flag1, desc) VALUES (?, ?, ?, ?)', (dbstruct.orgImage, flag0, flag1, pickle.dumps(dbstruct.features)))
            img_id = self.cursor.lastrowid  
            for obj in dbstruct.classes:
                self.cursor.execute('INSERT INTO objects (image_id, class_name, desc,weight) VALUES (?, ?, ?,?)', (img_id, obj.className, pickle.dumps(obj.features),obj.weight))
            self.con.commit()                
        except sqlite3.Error as e:
            self.con.rollback()
            print(f'Error occurred: {e}')

    def searchImageByTerm(self, termName) -> list[ImageAnalyzation.ImageData]:
            # img.*,
            self.cursor.execute("""
                SELECT o.*, i.* FROM inverted_index i
                JOIN objects o ON o.id = i.object_id
                JOIN terms t ON i.term_id = t.id
                JOIN images img ON img.id = o.image_id
                WHERE t.term = ?
            """, (termName,))

            #row[0] obj_id, img_id, obj_features, img_id, img_path, img_feautres
            start=time.time()
            rows = self.cursor.fetchall()#one
            image_objects  = dict()
            for row in rows:
                img_id = row[1]
                if img_id not in image_objects:
                    image_objects[img_id] = []
                
                image_objects[img_id].append(ImageAnalyzation.ImageClassificationData(termName, None, pickle.loads(row[2]),row[3]))
            print(time.time()-start)
            results: list[ImageAnalyzation.ImageData] = []
            for img_id in image_objects.keys():
                self.cursor.execute("SELECT i.* FROM images i WHERE i.id = ?", (img_id,))
                image = self.cursor.fetchone()
                results.append(ImageAnalyzation.ImageData(image[1], image_objects[img_id], pickle.loads(image[2])))
            return results#[DBStruct(termName, x[1], pickle.loads(x[2])) for x in rows]
        
        
    def search_by_image(self, terms:list):
        flag0, flag1 = get_image_flag(terms)
        self.cursor.execute("""
                SELECT i.*, o.* FROM images i
                JOIN objects o ON i.id = o.image_id
                WHERE ((i.flag0 & ?) != 0 AND (i.flag1 & ?) != 0) 
                OR ((i.flag0 & ?) != 0 AND ? = 0)
                OR (? = 0 AND (i.flag1 & ?) != 0)
            """, (flag0, flag1, flag0, flag1, flag0, flag1))
        rows = self.cursor.fetchall()        
        
        image_objects  = {}
        
        for row in rows:
            img_id, img_path, flag0, flag1, img_features, obj_id, _, class_name, obj_features, weight= row
            if img_id not in image_objects:
                image_objects[img_id] = ImageAnalyzation.ImageData(img_path, [], pickle.loads(img_features))
                
            image_objects[img_id].classes.append(ImageAnalyzation.ImageClassificationData(class_name,None,pickle.loads(obj_features),weight))
        return image_objects.values()#[DBStruct(termName, x[1], pickle.loads(x[2])) for x in rows]
    
    def close_connection(self):
        self.cursor.close()
        self.con.close()