import cv2
import csv
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import os
import PIL
from PIL import Image
from prettytable import PrettyTable
import random
from tqdm import tqdm
import tensorflow as tf
from logzero import logger
from ephem import readtle, degree
from exif import Image as exifImage
import reverse_geocoder as rg
from pathlib import Path
from time import sleep
from datetime import datetime

# Pre-requesite #
# Data basepath
base_path = os.getcwd() + '/Dataset'
dir_path = Path(__file__).parent.resolve()
data_file = dir_path/'data.csv'
# Ephem ISS location
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   20316.41516162  .00001589  00000+0  36499-4 0  9995"
line2 = "2 25544  51.6454 339.9628 0001882  94.8340 265.2864 15.49409479254842"
iss = readtle(name, line1, line2)
iss.compute()
print(iss.sublat, iss.sublong)


# Camera Resolution
'''
cam = PiCamera()
cam.resolution = (1296,972) # Valid resolution for V1 camera
This parameter play a major role in overall processing time
Capturing smaller images would allow not to scale them down during pre-processing
'''

# Pre-Processing Helper functions

def take_photo():
    """
    This requires the picamera library
    This function would replace get_data
    """

def capture(camera, image):
    """Use 'camera' to capture an 'image' file with lat/long EXIF data."""
    iss.compute() # Get the lat/long values from ephem

    # convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(iss.sublat)
    west, exif_longitude = convert(iss.sublong)

    # set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

    # capture the image
    camera.capture(image)

# ISS Location Helper functions
# https://support.google.com/maps/answer/18539?co=GENIE.Platform%3DDesktop&hl=fr
def convert(angle):
    """
    Convert an ephem angle (degrees:minutes:seconds) to
    an EXIF-appropriate representation (rationals)
    e.g. '51:35:19.7' to '51/1,35/1,197/10'
    Return a tuple containing a boolean and the converted angle,
    with the boolean indicating if the angle is negative.
    """
    degrees, minutes, seconds = (float(field) for field in str(angle).split(":"))
    exif_angle = f'{abs(degrees):.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return degrees < 0, exif_angle

    '''
    south, exif_latitude = convert(iss.sublat)
    west, exif_longitude = convert(iss.sublong)
    print(south,exif_latitude)
    print(west,exif_longitude)
    '''

def get_img_exif(img_name,pred,iss_var):
    iss_var.compute()
    exif_dico = {"Date/Time":datetime.now(), "Location":(iss_var.sublat,iss_var.sublong), "ImgName":img_name,"NO2":pred}
    return exif_dico


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
    Create and downsize NDVI images to decrease Training/Inference time
    """
    ndvi_img_list = []
    for i in tqdm(range(0,len(image_list))):
        img_path = os.path.join(base_path,image_list[i])
        ndvi_img = get_ndvi(img_path)
        ndvi_img = scale_down(ndvi_img)
        ndvi_img_list.append(ndvi_img)
    return ndvi_img_list

def ndvi_rgb_image(base_path,image_list):
    ndvi_rgb_list = []
    ndvi_images = ndvi_small_image(base_path,image_list)
    for i in range(0,len(image_list)):
        ndvi_rgb = cv2.cvtColor(ndvi_images[i],cv2.COLOR_GRAY2RGB)
        ndvi_rgb_list.append(ndvi_rgb)
    ndvi_rgb_numpy = np.array(ndvi_rgb_list)
    return ndvi_rgb_numpy


# Model Helper functions
def load_model(model_name):
    """
    Load a pre-trained model
    """
    model = tf.keras.models.load_model(model_name)
    return model

def make_inference(ndvi_image,model):
    """
    Make inference using model to get N02 predictions from 1 x NDVI image
    """
    ndvi_image_exp = np.expand_dims(ndvi_image,axis=0)
    prediction = model.predict(ndvi_image_exp)
    return prediction

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
    start_time = datetime.now()
    logger.info(f'Starting Bergson Astro Pi team experiment at {start_time}')
    
    
    # Datasets to Predict on - 20s on MAC for 82 images
    logger.info("Creating Datasets")
    image_list = get_data('./Dataset')


    # Exif Metadata
    '''
    img0_path = base_path+"/"+image_list[0]
    with open(img0_path, 'rb') as image_file:
        my_image = Image(image_file)    
        print(my_image.has_exif)
    
    exif_data = "test"
    im.save('imageExif.jpg',  exif=exif_data)
    '''
   
    # NDVI Preprocessing
    x_train_2D = np.array(ndvi_small_image(base_path,image_list),dtype=np.uint8)
    x_train_2D = np.expand_dims(x_train_2D,axis=3)
    img0_1c = x_train_2D[0]
    img0_3c = cv2.cvtColor(img0_1c,cv2.COLOR_GRAY2RGB)
    cv2.imwrite('test_ndvi_saving.jpeg',img0_3c)
    #img = Image.fromarray(x_train_2D[0])
    #img.save('test_ndvi_pil.jpeg')
    '''
    plt.axis('off')
    plt.imshow(x_train_2D[0])
    plt.savefig("test_ndvi_saving.jpeg")
    plt.show()
    '''

    # NDVI Images Analysis
    img0 = x_train_2D[0]
    print(type(img0))
    print(img0.shape)
    print(x_train_2D.dtype)
    img0

    # https://stackoverflow.com/questions/47438654/single-channel-png-displayed-with-colors
    # http://www.greensightag.com/logbook/dynamic-colorization-a-deeper-look-into-what-it-means/

    # Loading Model
    logger.info("Loading AI Convolutional Model")
    conv2D_model = load_model("conv2D.h5")
    print(conv2D_model.summary())

    # Doing Inference - 8s on MAC for 82 images
    logger.info("Doing Inference with AI Model")
    predictions = []
    for i in tqdm(range(0,len(image_list))):
        prediction = make_inference(x_train_2D[i],conv2D_model)
        predictions.append(prediction)

    # Get Decoded Inference results
    decoded_inference = decode_inference(predictions)
    print(decoded_inference)

    # Write Prediction as CSV to disk
    logger.info("Saving NO2 prediction from AI")
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ("Date/time", "Location", "Picture Name","Predicted NO2")
        writer.writerow(header)
        for i in range(0,len(decoded_inference)):
            exif_data = get_img_exif(image_list[i],decoded_inference[i],iss)
            row = (exif_data['Date/Time'], exif_data['Location'], exif_data['ImgName'],exif_data['NO2'])
            writer.writerow(row)
            sleep(0.5)

    rgb_ndvi_list = ndvi_rgb_image(base_path,image_list)

    # Write NDVI Image as JPG to disk with matplotlib
    logger.info("Saving Images from ISS")
    for i in tqdm(range(0,len(rgb_ndvi_list))):
        matplotlib.image.imsave(image_list[i]+"_ndvi"+".jpeg", rgb_ndvi_list[i])

    # End of experiment
    end_time = datetime.now()
    logger.info(f'Finishing Bergson Astro Pi team experiment at {end_time}')
    experiment_time = end_time - start_time
    logger.info(f'Bergson Astro Pi team experiment run time {experiment_time} for {len(image_list)} images')

    return decoded_inference
    
# Executing main
predictions = main()