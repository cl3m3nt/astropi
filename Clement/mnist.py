import tensorflow as tf
import tensorflow.keras.datasets as datasets
import matplotlib.pyplot as plt

# Get MNIST data
mnist = datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
print(x_train.shape)
print(y_train.shape)

# Visualize MNIST Data
for i in range(0,5):
    plt.imshow(x_train[i])
    plt.title(f'label:{y_train[i]}')
    plt.show()

# Get Model