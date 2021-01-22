import cv2
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from logzero import logger
from ephem import readtle, degree
from exif import Image as exifImage
import reverse_geocoder as rg
from pathlib import Path
from time import sleep
from datetime import datetime,timedelta
from picamera import PiCamera
from keras.models import model_from_json

# Data basepath
dir_path = Path(__file__).parent.resolve()
data_file = dir_path/'data.csv'

# Ephem ISS location
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   20316.41516162  .00001589  00000+0  36499-4 0  9995"
line2 = "2 25544  51.6454 339.9628 0001882  94.8340 265.2864 15.49409479254842"
iss = readtle(name, line1, line2)
iss.compute()

# Camera Resolution
rpi_cam = PiCamera()
rpi_cam.resolution = (648,486) #Using AI expected resolution

# Pre-Processing Helper functions

def capture(camera, image):
    """
    Use 'camera' to capture an 'image' file with lat/long EXIF data.
    """
    iss.compute() # Get the lat/long values from ephem

    # convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(iss.sublat)
    west, exif_longitude = convert(iss.sublong)

    # set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

    # capture the image to disk
    camera.capture(image)

# ISS Location Helper functions
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

def get_img_exif(img_name,iss_var,pred):
    iss_var.compute()
    exif_dico = {"Date/Time":datetime.now(), "Location":(iss_var.sublat,iss_var.sublong), "ImgName":img_name,"NO2":pred}
    return exif_dico


# Model Helper functions
def load_model(model_name):
    json_file = open(f'{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f'{model_name}.h5')
    print("Loaded model from disk")
    return loaded_model

'''
def load_model(model_name):
    """
    Load a pre-trained model
    """
    model = tf.keras.models.load_model(model_name)
    return model
'''

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


# Main function
def main():
    """
    Main program process of Bergson Astro Pi team
    """
    #import os
    #os.chdir('./Clement')

    # Experiment Start
    start_time = datetime.now()
    logger.info(f'Starting Bergson Astro Pi team experiment at {start_time}')

    # Load AI Model
    logger.info("Loading AI Convolutional Model")
    conv2D_model = load_model("model114")
    print(conv2D_model.summary())

    # Create Log File
    logger.info("Creating Log file")
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ("Date/time", "Location", "Picture Name","Predicted NO2")
        writer.writerow(header)


    # Start Loop over 3 hours

    now_time = datetime.now()
    # run a loop for 2 minutes
    while (now_time < start_time + timedelta(seconds=10)):

        # Take Earth Picture
        logger.info("Taking Picture from ISS")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        pic_name = f'bergson_img_{timestamp}.jpg'
        capture(rpi_cam,str(dir_path/pic_name))

        # NDVI Preprocessing
        logger.info("NDVI preprocessing")
        ndvi_image = get_ndvi(str(dir_path/pic_name))
        ndvi_image = np.expand_dims(ndvi_image,axis=2)

        # Do Inference with AI Model
        logger.info("Doing Inference with AI Model")
        prediction = make_inference(ndvi_image,conv2D_model)
        
        # Get Decoded Inference results
        logger.info("Decoding Inference to NO2 prediction")
        decoded_prediction = decode_prediction(prediction)
        logger.info(decoded_prediction)

        # Write Prediction as CSV to disk
        logger.info("Saving NO2 prediction to Log File")
        exif_data = get_img_exif(pic_name,iss,decoded_prediction)
        row = (exif_data['Date/Time'], exif_data['Location'], pic_name,exif_data['NO2'])
        with open(data_file,mode='a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Write NDVI Image as JPG to disk with matplotlib
        '''
        logger.info("Saving NDVI Images from Experiment")
        matplotlib.image.imsave(f'bergson_img_{timestamp}_ndvi.jpeg', ndvi_image)
        '''

        # update the current time
        now_time = datetime.now()

    # End Loop over 3 hours


    # Experiment End
    end_time = datetime.now()
    logger.info(f'Finishing Bergson Astro Pi team experiment at {end_time}')
    experiment_time = end_time - start_time
    logger.info(f'Bergson Astro Pi team experiment run time {experiment_time}')

# Executing main
main()
