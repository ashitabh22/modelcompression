from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import save_model, load_model

def cifar_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = x_train.shape[1:]
    num_classes =10

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train,y_train,x_test,y_test


# model = load_model("./best_squeezenet_del.h5")

# model.summary()
# loss,acc = model.evaluate(x_test,y_test)
# print(f"Accuracy: {acc}")




