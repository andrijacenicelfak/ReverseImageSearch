import sqlite3
import pickle
import time
from ImageAnalyzationModule.ImageAnalyzationDataTypes import ImageData, ImageClassificationData, BoundingBox
from DB.Functions import *

def decode_image_flag(flag):
    decoded_terms = []
    for term, bit_position in model_names.items():
        if flag & (2 ** bit_position):
            decoded_terms.append(term)
    return decoded_terms
  
class ImageDB:
    def __init__(self):
        self.cursor=""

    def open_connection(self):
        self.con = sqlite3.connect("database.db")
        self.cursor = self.con.cursor()
        try:
            self.con = sqlite3.connect("database.db")
            self.cursor = self.con.cursor()
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    id INTEGER PRIMARY KEY,
                    image_id INT,
                    class_name TEXT NOT NULL,
                    desc BLOB NOT NULL,
                    weight REAL NOT NULL,
                    conf REAL NOT NULL,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                );
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS images(
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL,
                    flag0 INTEGER NOT NULL,
                    flag1 INTEGER NOT NULL,
                    desc BLOB NOT NULL,
                    caption_vec BLOB NOT NULL,
                    caption TEXT NOT NULL
                );
            """)        
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_objects_image_id ON objects(image_id);")    
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_flag ON images(flag0,flag1);")       
   
            self.con.commit()
        except sqlite3.Error as e:
            print(f'Error occurred: {e}')

        
    def addImage(self,dbstruct: ImageData,commit_flag=True):#orgImage is image but here is path lol
        try:
            flag0, flag1 = get_image_flag([x.className for x in dbstruct.classes])
            self.cursor.execute('INSERT INTO images (path, flag0, flag1, desc, caption_vec, caption) VALUES (?, ?, ?, ?, ?, ?)', (dbstruct.orgImage, flag0, flag1, pickle.dumps(dbstruct.features), pickle.dumps(dbstruct.vector), dbstruct.description))
            img_id = self.cursor.lastrowid  
            for obj in dbstruct.classes:
                self.cursor.execute('INSERT INTO objects (image_id, class_name, desc,weight, conf) VALUES (?, ?, ?,?, ?)', (img_id, obj.className, pickle.dumps(obj.features),obj.weight, obj.conf))
            if commit_flag:
                self.con.commit()                
        except sqlite3.Error as e:
            self.con.rollback()
            print(f'Error occurred: {e}')
    
    def commit_changes(self):
        self.con.commit()
    
    def searchImageByTerm(self, termName) -> list[ ImageData]:
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
                
                image_objects[img_id].append(ImageClassificationData(termName, None, pickle.loads(row[2]),row[3], row[4]))
            print(time.time()-start)
            results: list[ImageData] = []
            for img_id in image_objects.keys():
                self.cursor.execute("SELECT i.* FROM images i WHERE i.id = ?", (img_id,))
                image = self.cursor.fetchone()
                results.append(ImageData(image[1], image_objects[img_id], pickle.loads(image[2]), pickle.loads(image[3]), image[4]))
            return results#[DBStruct(termName, x[1], pickle.loads(x[2])) for x in rows]
        
        
    def search_by_image(self, terms:list):
        flag0, flag1 = get_image_flag(terms)
        
        if flag0 != 0 or flag1 != 0:
            self.cursor.execute("""
                    SELECT i.*, o.* FROM images i
                    JOIN objects o ON i.id = o.image_id
                    WHERE ((i.flag0 & ?) != 0 AND (i.flag1 & ?) != 0) 
                    OR ((i.flag0 & ?) != 0 AND ? = 0)
                    OR (? = 0 AND (i.flag1 & ?) != 0)
                """, (flag0, flag1, flag0, flag1, flag0, flag1))
        else:
            self.cursor.execute("""
                    SELECT i.*, 0.* FROM images i
                    JOIN objects o ON i.id = o.image_id
                    WHERE i.flag0 = 0 AND i.flag1 = 0
                    """)       
            
        rows = self.cursor.fetchall()        
        
        image_objects  = {}
        
        for row in rows:
            img_id, img_path, flag0, flag1, img_features, caption_vec, caption, obj_id, _, class_name, obj_features, weight, conf= row
            if img_id not in image_objects:
                image_objects[img_id] = ImageData(img_path, [], pickle.loads(img_features), pickle.loads(caption_vec), caption)
                
            image_objects[img_id].classes.append(ImageClassificationData(class_name,None,pickle.loads(obj_features),weight, conf=conf))
        return image_objects.values()#[DBStruct(termName, x[1], pickle.loads(x[2])) for x in rows]
    
    def close_connection(self):
        self.cursor.close()
        self.con.close()
        print("Closed connection!")