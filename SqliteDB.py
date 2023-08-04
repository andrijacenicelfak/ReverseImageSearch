import sqlite3
import pickle
import time
import cv2

from ImageAnalyzation import ImageClassificationData, ImageData
# class DBObjectStruct:
#     def __init__(self, term, vec):
#         self.term = term
#         self.vector = vec

# class DBStruct:
#     def __init__(self, path, vector, objects:list[DBObjectStruct]) -> None:
#         #slika
#         self.path = path
#         self.vector = vector
#         self.objects = objects

class ImageDB:
    def __init__(self):
        print()
    def open_connection(self):
        self.con = sqlite3.connect("test2.db")
        self.cursor = self.con.cursor()
        try:
            self.con = sqlite3.connect("test2.db")
            self.cursor = self.con.cursor()
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    id INTEGER PRIMARY KEY,
                    image_id INT,
                    desc BLOB NOT NULL,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                );
            """)
            
            self.cursor.execute("""  CREATE TABLE IF NOT EXISTS terms (
                                id INTEGER PRIMARY KEY,
                                term TEXT UNIQUE
                );
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS inverted_index(
                    term_id INT,
                    object_id INT,
                    PRIMARY KEY (term_id, object_id),
                    FOREIGN KEY (term_id) REFERENCES terms(id),
                    FOREIGN KEY (object_id) REFERENCES objects(id)                    
                );
            """)

            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS images(
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL,
                    desc BLOB NOT NULL         
                );
            """)               
            self.con.commit()
        except sqlite3.Error as e:
            print(f'Error occurred: {e}')

        
    def addImage(self,dbstruct:ImageData):#orgImage is image but here is path lol
        try:
            self.cursor.execute('INSERT INTO images (path, desc) VALUES (?, ?)', (dbstruct.orgImage, pickle.dumps(dbstruct.features)))
            img_id = self.cursor.lastrowid  
            for obj in dbstruct.classes:
               
                self.cursor.execute("SELECT * FROM terms WHERE term = ?", (obj.className,))
                term = self.cursor.fetchone()
                if term == None:
                    self.cursor.execute(f"""INSERT INTO terms (term) VALUES (?)
                    """, (obj.className,))
                    term_id = self.cursor.lastrowid
                else:
                    term_id = term[0]

                self.cursor.execute('INSERT INTO objects (image_id, desc) VALUES (?, ?)', (img_id, pickle.dumps(obj.features)))
                obj_id = self.cursor.lastrowid

                self.cursor.execute('INSERT INTO inverted_index (term_id, object_id) VALUES (?, ?)', (term_id, obj_id))
                
                self.con.commit()
            
        except sqlite3.Error as e:
            print(f'Error occurred: {e}')

    def searchImageByTerm(self, termName) -> list[ImageData]:
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
                
                image_objects[img_id].append(ImageClassificationData(termName, None, pickle.loads(row[2])))
            print(time.time()-start)
            results: list[ImageData] = []
            for img_id in image_objects.keys():
                self.cursor.execute("SELECT i.* FROM images i WHERE i.id = ?", (img_id,))
                image = self.cursor.fetchone()
                results.append(ImageData(image[1], image_objects[img_id], pickle.loads(image[2])))
            return results#[DBStruct(termName, x[1], pickle.loads(x[2])) for x in rows]
            
    def close(self):
        self.cursor.close()
        self.con.close()

#razlka histograma, -> klasteriyacija