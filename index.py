import os
import cv2
import sys
from sqliteDB import ImageDB
from ImageAnalyzation import ImageAnalyzation
def read_images_from_dir(dir_path):
    file_list = os.listdir(dir_path)
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(dir_path, file_name)
            image = cv2.imread(image_path)       
            if image is not None:
                yield (image_path,image)
            else:
                print(f"Unable to read image: {file_name}")