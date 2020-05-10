from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation
from keras.layers.core import Dense
import tensorflow as tf
import keras.backend as K


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def get_model():
    MLP_Mnist_model = Sequential()
    MLP_Mnist_model.add(Dense(512, input_shape=(784,)))
    MLP_Mnist_model.add(Activation('relu'))
    MLP_Mnist_model.add(Dropout(0.2))

    MLP_Mnist_model.add(Dense(512))
    MLP_Mnist_model.add(Activation('relu'))
    MLP_Mnist_model.add(Dropout(0.2))

    MLP_Mnist_model.add(Dense(10))
    MLP_Mnist_model.add(Activation('softmax'))

    MLP_Mnist_model.compile(loss='categorical_crossentropy', metrics=['accuracy', f1], optimizer='adam')
    return MLP_Mnist_model
