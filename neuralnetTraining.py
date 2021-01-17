import cv2
import matplotlib.pyplot as plt 
import numpy as np
import os
from PIL import Image
from prettytable import PrettyTable
import random
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


def get_conv1D_model():
    model = Sequential([
    tf.keras.layers.Conv1D(8,(3),input_shape=(1944,2592)),
    tf.keras.layers.MaxPool1D(2,2),
    tf.keras.layers.Conv1D(16,(3)),
    tf.keras.layers.MaxPool1D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3,activation='softmax')
])
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

    model.summary()
    return model


def get_conv2D_model():
    model = Sequential([
    tf.keras.layers.Conv2D(8,(3,3),input_shape=(486,648,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(16,(3,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3,activation='softmax')
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
    mobilenetv2_preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    input_shape = (486,648,3)
    img_in = Input(shape=input_shape, name='img_in')
    x = mobilenetv2_preprocess(img_in)
    x = mobilenetv2(img_in, training=True)
    x = GlobalAveragePooling2D()(x)
    # Classification layer
    output = Dense(3, activation='softmax', name='dense')(x)
    # Final model
    model = Model(inputs=[img_in], outputs=output)

    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

    model.summary()
    return model

def get_efficientnetb0_model():
    efficientnetb0 = EfficientNetB0(include_top=False, weights='imagenet',input_shape=(486,648,3))
    for layer in efficientnetb0.layers:
        layer.trainable = False
    efficientnetb0_preprocess = tf.keras.applications.efficientnet.preprocess_input

    # Transfer Learning
    input_shape = (486,648,3)
    img_in = Input(shape=input_shape, name='img_in')
    x = efficientnetb0_preprocess(img_in)
    x = efficientnetb0(img_in, training=True)
    x = GlobalAveragePooling2D()(x)
    # Classification layer
    output = Dense(3, activation='softmax', name='dense')(x)
    # Final model
    model = Model(inputs=[img_in], outputs=output)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
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


def save_model(model_name,model):
    model.save(model_name + ".h5")
    print(f"Model saved as {model_name}.h5")

# Conv1D Training
logger.info("Conv1D Training Start")
x_train = ndvi_images()
y_train = no2_labels()
conv1D_model = get_conv1D_model()
history_conv1D = conv1D_model.fit(x_train,y_train,epochs=5)
save_model("conv1D",conv1D_model)

# Conv2D Training
logger.info("Conv2D Training Start")
x_train = np.array(ndvi_small_image())
x_train = np.expand_dims(x_train,axis=3)
y_train = no2_labels()
conv2D_model = get_conv2D_model()
history_conv2D = conv2D_model.fit(x_train,y_train,epochs=5)
save_model("conv2D",conv2D_model)

# Mobilenetv2 Training
logger.info("Mobilenetv2 Training Start")
print('x_train for mobilenet')
x_train = ndvi_rgb_image()
y_train = no2_labels()
print(' transfer mobilenet')
mobilenetv2_model = get_mobilenetv2_model()
history_mobilenetv2 = mobilenetv2_model.fit(x_train,y_train,epochs=5)
save_model("mobilenetv2",mobilenetv2_model)

# EfficientB0 Training
logger.info("EfficientNetB0 Training Start")
x_train = ndvi_rgb_image()
y_train = no2_labels()
efficientnetb0_model = get_efficientnetb0_model()
history_efficientnetb0 = efficientnetb0_model.fit(x_train,y_train,epochs=5)
save_model("efficientnetb0",efficientnetb0_model)

# Training Models Metrics
metrics = PrettyTable()
metrics.field_names = ["Model","Total Params","Trainable Params","Non-Trainable Params","Loss", "Accuracy"]
metrics.add_row(["Conv1D",get_model_params(conv1D_model)[0],get_model_params(conv1D_model)[1],get_model_params(conv1D_model)[2], history_conv1D.history['loss'], history_conv1D.history['accuracy']])
metrics.add_row(["Conv2D",get_model_params(conv2D_model)[0],get_model_params(conv2D_model)[1],get_model_params(conv2D_model)[2], history_conv2D.history['loss'], history_conv2D.history['accuracy']])
metrics.add_row(["MobileNetv2",get_model_params(mobilenetv2_model)[0],get_model_params(mobilenetv2_model)[1],get_model_params(mobilenetv2_model)[2],history_mobilenetv2.history['loss'], history_mobilenetv2.history['accuracy']])
metrics.add_row(["EfficientNetB0",get_model_params(efficientnetb0_model)[0],get_model_params(efficientnetb0_model)[1],get_model_params(efficientnetb0_model)[2], history_efficientnetb0.history['loss'], history_efficientnetb0.history['accuracy']])
print(metrics)
