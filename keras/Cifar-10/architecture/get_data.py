import keras
from keras.datasets import mnist

def get_data_reshape():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(len(x_train), 784)
    x_train = x_train.astype('float32')/255
    x_test = x_test.reshape(len(x_test), 784)
    x_test = x_test.astype('float32')/255   
    # convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test
