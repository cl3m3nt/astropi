import cv2
import matplotlib.pyplot as plt 
import numpy as np
import os
from PIL import Image
from prettytable import PrettyTable
import random
from tqdm import tqdm
import tensorflow as tf
import logging

# Pre-requesite
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
base_path = os.getcwd() + '/Dataset'


# Pre-Processing Helper functions

def get_data(images_path):
    """
    Get Input Data to be pre-processed
    """
    images_list = os.listdir(images_path)
    return images_list

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

def ndvi_small_image(base_path,image_list):
    """
    Downsize NDVI images to decrease Training/Inference time
    """
    ndvi_img_list = []
    for i in tqdm(range(0,len(image_list))):
        img_path = os.path.join(base_path,image_list[i])
        ndvi_img = get_ndvi(img_path)
        ndvi_img = scale_down(ndvi_img)
        ndvi_img_list.append(ndvi_img)
    return ndvi_img_list


# Model Helper functions
def load_model(model_name):
    """
    Load a pre-trained model
    """
    model = tf.keras.models.load_model(model_name)
    return model

def make_inference(x_train_2D,conv2D_model):
    """
    Make inference using model to get N02 predictions from NDVI images
    """
    x_train = []
    predictions = []
    for i in range(0,82):
        x_train_i = np.expand_dims(x_train_2D[i],axis=0)
        x_train.append(x_train_i)
    for i in tqdm(range(0,82)):
        prediction = conv2D_model.predict(x_train[i])
        predictions.append(prediction)
    return predictions

def decode_prediction(prediction):
    """
    Decode 3-value float prediction to a string value among "low","medium","high"
    """
    no2_level = ["low","medium","high"]
    no2_max = np.argmax(prediction)
    no2_prediction = no2_level[no2_max]
    return no2_prediction

def decode_inference(predictions):
    decoded_inference = []
    for prediction in tqdm(predictions):
        decoded_prediction = decode_prediction(prediction)
        decoded_inference.append(decoded_prediction)
    return decoded_inference


# Main function
def main():
    """
    Main program process of Bergson Astro Pi team
    """

    # Datasets to Predict on - 20s on MAC for 82 images
    logger.info("Creating Datasets")
    image_list = get_data('./Dataset')
    x_train_2D = np.array(ndvi_small_image(base_path,image_list))
    x_train_2D = np.expand_dims(x_train_2D,axis=3)

    # Loading Model
    logger.info("Loading Models")
    conv2D_model = load_model("conv2D.h5")
    print(conv2D_model.summary())

    # Doing Inference - 8s on MAC for 82 images
    logger.info("Doing Inference")
    predictions = make_inference(x_train_2D,conv2D_model)
    print(predictions)

    # Get Decoded Inference results
    decoded_inference = decode_inference(predictions)
    print(decoded_inference)
    
# Executing main
main()