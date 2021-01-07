import cv2
import matplotlib.pyplot as plt 
import numpy as np
import os
from PIL import Image
from prettytable import PrettyTable
import random
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Get Data
base_path = os.getcwd() + '/Dataset'
image_list = os.listdir('./Dataset')

def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-100%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 100)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out


def get_ndvi(image_path):
    """
    Transform a raw image to ndvi image
    """
    image = cv2.imread(image_path) 
    b, g, r = cv2.split(image)

    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.00001  # Make sure we don't divide by zero!
    ndvi_image = (r.astype(float) - b) / bottom
    ndvi_image = contrast_stretch(ndvi_image)
    ndvi_image = ndvi_image.astype(np.uint8)
    return ndvi_image

def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-100%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 100)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out


def get_ndvi(image_path):
    """
    Transform a raw image to ndvi image
    """
    image = cv2.imread(image_path) 
    b, g, r = cv2.split(image)

    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.00001  # Make sure we don't divide by zero!
    ndvi_image = (r.astype(float) - b) / bottom
    ndvi_image = contrast_stretch(ndvi_image)
    ndvi_image = ndvi_image.astype(np.uint8)
    return ndvi_image


def get_ndvi(image_path):
    """
    Transform a raw image to ndvi image
    """
    image = cv2.imread(image_path) 
    b, g, r = cv2.split(image)

    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.00001  # Make sure we don't divide by zero!
    ndvi_image = (r.astype(float) - b) / bottom
    ndvi_image = contrast_stretch(ndvi_image)
    ndvi_image = ndvi_image.astype(np.uint8)
    return ndvi_image


def ndvi_images():
    ndvi_img_list = []
    for i in range(0,len(image_list)):
        img_path = os.path.join(base_path,image_list[i])
        ndvi_img = get_ndvi(img_path)
        ndvi_img_list.append(ndvi_img)
    ndvi_img_numpy = np.array(ndvi_img_list)
    return ndvi_img_numpy

def ndvi_small_image():
    ndvi_img_list = []
    for i in range(0,len(image_list)):
        img_path = os.path.join(base_path,image_list[i])
        ndvi_img = get_ndvi(img_path)
        ndvi_img = scale_down(ndvi_img)
        ndvi_img_list.append(ndvi_img)
    return ndvi_img_list

def ndvi_rgb_image():
    ndvi_rgb_list = []
    ndvi_images = ndvi_small_image()
    for i in range(0,len(image_list)):
        ndvi_rgb = cv2.cvtColor(ndvi_images[i],cv2.COLOR_GRAY2RGB)
        ndvi_rgb_list.append(ndvi_rgb)
    ndvi_rgb_numpy = np.array(ndvi_rgb_list)
    return ndvi_rgb_numpy

def no2_labels():
    label_list = []
    for i in range(0,82):
        random_label = random.randint(0,2)
        label_list.append(random_label)
    label_numpy = np.array(label_list)
    return label_numpy


def scale_down(image):
    src = image
    #percent by which the image is resized
    scale_percent = 25

    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)
    return output

# Get NDVI images
def ndvi_images():
    ndvi_img_list = []
    for i in range(0,len(image_list)):
        img_path = os.path.join(base_path,image_list[i])
        ndvi_img = get_ndvi(img_path)
        ndvi_img_list.append(ndvi_img)
    ndvi_img_numpy = np.array(ndvi_img_list)
    return ndvi_img_numpy

# Get NO2 label within [0,2]
def no2_labels():
    label_list = []
    for i in range(0,82):
        random_label = random.randint(0,2)
        label_list.append(random_label)
    label_numpy = np.array(label_list)
    return label_numpy



# Model Helper functions
def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model

def get_model_params(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    '''
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    '''
    return (trainable_count + non_trainable_count),trainable_count,non_trainable_count

# Datasets
logger.info("Creating Datasets")
x_train_1D = ndvi_images()
y_train_1D = no2_labels()
x_train_2D = np.array(ndvi_small_image())
x_train_2D = np.expand_dims(x_train_2D,axis=3)
y_train_2D = no2_labels()
x_train_transfer = ndvi_rgb_image()
y_train_transfer = no2_labels()


# Loading model
logger.info("Loading Models")
conv1D_model = load_model("conv1D.h5")
conv2D_model = load_model("conv2D.h5")
mobilenetv2_model = load_model("mobilenetv2.h5")
efficientnetb0_model = load_model("efficientnetB0.h5")
models_list = [conv1D_model,conv2D_model,mobilenetv2_model,efficientnetb0_model]

# Get Model info
print(conv1D_model.summary())
print(conv2D_model.summary())
print(mobilenetv2_model.summary())
print(efficientnetb0_model.summary())

# Model Inference
logger.info("Doing Inference")

logger.info("Conv1D model Inference")
x_train = []
for i in range(0,82):
    x_train_i = np.expand_dims(x_train_1D[i],axis=0)
    x_train.append(x_train_i)
for i in tqdm(range(0,82)):
    conv1D_model.predict(x_train[i])

logger.info("Conv2D model Inference")
x_train = []
for i in range(0,82):
    x_train_i = np.expand_dims(x_train_2D[i],axis=0)
    x_train.append(x_train_i)
for i in tqdm(range(0,82)):
    conv2D_model.predict(x_train[i])

logger.info("Mobilenetv2 model Inference")
x_train = []
for i in range(0,82):
    x_train_i = np.expand_dims(x_train_transfer[i],axis=0)
    x_train.append(x_train_i)
for i in tqdm(range(0,82)):
    mobilenetv2_model.predict(x_train[i])

logger.info("EfficientNetB0 model Inference")
x_train = []
for i in range(0,82):
    x_train_i = np.expand_dims(x_train_transfer[i],axis=0)
    x_train.append(x_train_i)
for i in tqdm(range(0,82)):
    efficientnetb0_model.predict(x_train[i])






