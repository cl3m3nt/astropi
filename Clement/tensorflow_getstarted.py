import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt

# Check TF version
print(tf.__version__)
print(tf.keras.__version__)

# Get Mnist Data
mnist = datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Visualize Data
plt.imshow(x_train[0])
plt.title('label:'+str(y_train[0]))
plt.show()

# Get MobilenetV2 model
model = MobileNetV2(weights='imagenet',include_top=False)
model.summary()