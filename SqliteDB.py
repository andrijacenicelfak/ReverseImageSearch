import sqlite3
import pickle
import cv2

class DBStruct:
    def __init__(self, term=None, path_to_image=None, descriptor=None) -> None:
        self.term = term
        self.path_to_image = path_to_image
        self.descriptor = descriptor

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
                    path TEXT NOT NULL,
                    desc TEXT NOT NULL
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
                    FOREIGN KEY (term_id) REFERENCES terms(term_id),
                    FOREIGN KEY (object_id) REFERENCES objects(object_id)                    
                );
            """)
                
            self.con.commit()
        except sqlite3.Error as e:
            print(f'Error occurred: {e}')
    def addImage(self,dbstruct:DBStruct):
        try:
            self.cursor.execute("SELECT * FROM terms WHERE term = ?", (dbstruct.term,))
            term = self.cursor.fetchone()
            if term == None:
                self.cursor.execute(f"""INSERT INTO terms (term) VALUES (?)
                """, (dbstruct.term,))
                term_id = self.cursor.lastrowid
            else:
                term_id = term[0]

            self.cursor.execute('INSERT INTO objects (path, desc) VALUES (?, ?)', (dbstruct.path_to_image, pickle.dumps(dbstruct.descriptor)))
            obj_id = self.cursor.lastrowid

            self.cursor.execute('INSERT INTO inverted_index (term_id, object_id) VALUES (?, ?)', (term_id, obj_id))
            
            self.con.commit()
            
        except sqlite3.Error as e:
            print(f'Error occurred: {e}')

    def searchImageByTerm(self, termName):
            self.cursor.execute("""
                SELECT o.*
                FROM objects o
                JOIN inverted_index i ON o.id = i.object_id
                JOIN terms t ON i.term_id = t.id
                WHERE t.term = ?
            """, (termName,))

            rows = self.cursor.fetchall()#one
            
            return [DBStruct(termName, x[1], pickle.loads(x[2])) for x in rows]
            
    def close(self):
        self.cursor.close()
        self.con.close()

#razlka histograma, -> klasteriyacija