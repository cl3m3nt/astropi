import cv2
import matplotlib.pyplot as plt 
import numpy as np
import os
from PIL import Image
from prettytable import PrettyTable
import random
import keras
from keras.models import Sequential, Model
from keras.layers import Input,Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense
from keras.applications import MobileNetV2
from keras.applications.mobilenet import preprocess_input
from keras.models import model_from_json
import logging
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Get Data
base_path = os.getcwd() + '/Dataset'
image_list = os.listdir('./Dataset')


# NDVI helper functions
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

# Artificial Intelligence Definition

def get_conv2D_model():
    model = keras.Sequential([
    keras.layers.Conv2D(8,(3,3),input_shape=(486,648,1),use_bias=False),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(16,(3,3),use_bias=False),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(3,activation='softmax')
])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def get_mobilenetv2_model():
    mobilenetv2 = MobileNetV2(include_top=False, weights='imagenet',input_shape=(486,648,3))
    for layer in mobilenetv2.layers:
        layer.trainable = False
    mobilenetv2_preprocess = preprocess_input
    input_shape = (486,648,3)
    input = keras.layers.Input(shape=input_shape, name='img_in')
    #x = mobilenetv2_preprocess(input)
    x = mobilenetv2(input)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # Classification layer
    output = keras.layers.Dense(3, activation='softmax', name='dense')(x)
    # Final model
    model = Model(inputs=input, outputs=output)

    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

    model.summary()
    return model

def save_model(model_name,model):
    model.save(model_name + ".h5")
    print(f"Model saved as {model_name}.h5")

'''
# Conv2D Training
keras.backend.clear_session()
logger.info("Conv2D Training Start")
x_train = np.array(ndvi_small_image())
x_train = np.expand_dims(x_train,axis=3)
x_train = x_train/255.0
y_train = no2_labels()
conv2D_model = get_conv2D_model()
history_conv2D = conv2D_model.fit(x_train,y_train,epochs=10)
save_model("Conv2D_TFLite",conv2D_model)

# Keras Save
# serialize model to JSON
model_json = conv2D_model.to_json()
with open("Conv2D_TF114.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
conv2D_model.save_weights("Conv2D_TF114.h5")
print("Saved model to disk")

# Keras Load
# load json and create model
json_file = open('Conv2D_TF114.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Conv2D_TF114.h5")
print("Loaded model from disk")
print(loaded_model.summary())
'''

'''
# TFLite Convert
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("Conv2D_TFLite.h5")
tflite_model = converter.convert()
with open('Conv2D_TF114.tflite','wb') as f:
    f.write(tflite_model)

'''
# Mobilenetv2 Training
logger.info("Mobilenetv2 Training Start")
x_train = ndvi_rgb_image()
x_train = x_train/255.0
y_train = no2_labels()
print(' transfer mobilenet')
mobilenetv2_model = get_mobilenetv2_model()
history_mobilenetv2 = mobilenetv2_model.fit(x_train,y_train,epochs=1)
save_model("Mobilenetv2_TFLite",mobilenetv2_model)

# Keras Save
# serialize model to JSON
model_json = mobilenetv2_model.to_json()
with open("Mobilenetv2_TF114.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
mobilenetv2_model.save_weights("Mobilenetv2_TF114.h5")
print("Saved model to disk")

# Keras Load
# load json and create model
json_file = open('Mobilenetv2_TF114.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Mobilenetv2_TF114.h5")
print("Loaded model from disk")
print(loaded_model.summary())

# TFLite Convert
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("Mobilenetv2_TFLite.h5")
tflite_model = converter.convert()
with open('Mobilenetv2_TF114.tflite','wb') as f:
    f.write(tflite_model)
