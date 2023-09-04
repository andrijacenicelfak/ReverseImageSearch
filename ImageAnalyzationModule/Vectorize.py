import os
import gdown
import gensim
import numpy as np

TRAIN_FILE=r"model\dataset\train_coco_2022.txt"
TEST_FILE=r"model\dataset\test_coco_2022.txt"
MODEL_FOLDER="model"
URL="https://drive.google.com/drive/folders/18QYbANxavHPfpjm55xZy1POOyBb5vsfI"

def get_latest_model_path():
    subfolders = [f for f in os.listdir(MODEL_FOLDER) if os.path.isdir(os.path.join(MODEL_FOLDER, f))]
    if not subfolders:
        return None
    highest_number_folder = max((folder for folder in subfolders if folder != "dataset"),key=lambda x: int(x))
    highest_number_folder_path = os.path.join(MODEL_FOLDER, highest_number_folder)
    model_files = [f for f in os.listdir(highest_number_folder_path) if f.endswith('.model')]
    model_file_path = os.path.join(highest_number_folder_path, model_files[0])
    return model_file_path

def download_model():
    print("Downloading model...")
    gdown.download_folder(URL,quiet=True,use_cookies=False)#mora ovako umesto MODEL_FOLDER :'(
    print("Downloaded model!...")
    
def cosine_similarity(vec1,vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

class Vectorize:
    def __init__(self):
        if not os.path.isdir(MODEL_FOLDER):
            download_model()
        model_file=get_latest_model_path()
        self.model=gensim.models.doc2vec.Doc2Vec.load(model_file)
    def infer_vector(self,sentence):
        mid=gensim.utils.simple_preprocess(sentence)
        return self.model.infer_vector(mid)
    def compare_sentences(self,sentence1,sentence2):
        vec1=self.infer_vector(sentence1)
        vec2=self.infer_vector(sentence2)
        return cosine_similarity(vec1,vec2)
    def compare_vectors(self,vec1,vec2):
        return cosine_similarity(vec1,vec2)