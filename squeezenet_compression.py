import tensorflow as tf
import numpy as np
from callbacks import OrderBy_Callback,MyLogger_Class
from data_sn import cifar_data


if __name__ == "__main__":

    model = tf.keras.models.load_model('./squeezenet_cifar.h5')
    x_train,y_train,test_images,test_labels= cifar_data()

    model.fit(x_train,y_train,batch_size = 64,epochs =15,
              callbacks=[OrderBy_Callback(test_images = test_images,
                                          test_labels = test_labels,
                                          leniency=0.3,
                                          types_of_layers=["conv"])])




