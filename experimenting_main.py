import tensorflow as tf
import logging
from weights_classes import Weights_fc,Weights_conv,check_on_blocksize,setup_weights_forcompression
from data import get_data_mnist,get_data_cifar
import numpy as np
import os
from callbacks import OrderBy_Callback,MyLogger_Class,experiment_key
from callbacks import ResultSummary_CompressionStats,folder_name



if __name__ == "__main__":

    model = tf.keras.models.load_model('./base_models/alexnet_usethis_mnist.h5')





    mylogger = MyLogger_Class(heading = "######## Alexnet MNIST WALTing Compression\n",
                             experiment_key = experiment_key)
    ResultSummary_CompressionStats.info(mylogger.heading)
    ResultSummary_CompressionStats.info(mylogger.experiment_key)

    train_images,train_labels,test_images,test_labels  = get_data_mnist()

    # log the dataset numbers here, add later.
    total_num = 1002
    batch_size = 64
    num_of_epochs =2

    model.fit(train_images[:total_num],train_labels[:total_num],batch_size=batch_size,
                           epochs = num_of_epochs,callbacks=[OrderBy_Callback(test_images = test_images,
                                                                              test_labels = test_labels)])

    # print(folder_name)
    tf.keras.models.save_model(model,'./{folder_name}/models/final.h5')









