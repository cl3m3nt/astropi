from time import sleep
from picamera import PiCamera
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import csv
from datetime import datetime,timedelta
from random import random

#https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays

print("Program start successfully")

dir_path = Path(__file__).parent.resolve()
data_path = "/home/pi/Data"
img_name = "zz_astropi_1_photo_170.jpg"
img_path = os.path.join(data_path, img_name)
print(img_path)

img = Image.open(img_path)

def img_info(image):
    print(image.format)
    print(image.size)
    print(image.mode)

def get_numpy_img(image_path):
    np_img = plt.imread(image_path)
    return np_img

def show_np_img(np_img):
    plt.imshow(np_image)
    plt.show()

def get_ndvi(np_img):
    band5 = 2 * np_img #Near Infra Red (NIR)
    band4 = np_img # Vision (VIS)
    ndvi_image = (band5-band4) / (band5+band4)
    return ndvi_image

def ai_model(image):
    scale = 100
    model = scale * random()
    return model

def predict_NO2(image):
    no2_level = ai_model(image)
    return no2_level

def astropi_experiment(img_path):
    np_image = get_numpy_img(img_path)
    ndvi_image = get_ndvi(np_image)
    no2_prediction = predict_NO2(ndvi_image)
    return no2_prediction

def main():
    no2_predictions  = []
    for i in range(0,10):
        index_start = 170
        img_base = '/home/pi/Data/zz_astropi_1_photo_'
        img_name = img_base+str(i+170)+'.jpg'
        no2_prediction = astropi_experiment(img_name)
        no2_predictions.append(no2_prediction)
    print(no2_predictions)

main()

print("Program ended successfully")