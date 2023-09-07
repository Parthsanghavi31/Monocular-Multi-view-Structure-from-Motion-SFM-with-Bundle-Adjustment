import glob
import cv2
import csv
import numpy as np
import random
from IPython import embed

class DataUtils:
    def __init__(self, base_path):
        self.base_path = base_path

    def import_images(self):
        random.seed(40)
        img_files = glob.glob(f"{self.base_path}/*.png", recursive=False)

        img_names = []
        for img_file in img_files:
            img_name = img_file.rsplit(".", 1)[0][-1]
            img_names.append(img_name)
            
            imgs = {int(img_name) : cv2.imread(img_file) for img_file, img_name in zip(img_files,img_names)}
        return imgs, img_names
    
    def load_camera_instrinsics(self):
        K = []
        with open(self.base_path+'calibration.txt') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                row_K = [float(row[i]) for i in range(3)]
                K.append(row_K)
        return np.array(K)
        