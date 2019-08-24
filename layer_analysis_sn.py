import tensorflow as tf
import numpy as np
from data_sn import cifar_data
import operator
from conv_conversions import resize_to_2D,resize_to_4D


def total_size(list_of_arrays):
    total =0
    for r in list_of_arrays:
        total = total + r.nbytes
        # print(r.nbytes)
    return total/2**20
if __name__ == "__main__":
    model =tf.keras.models.load_model("./squeezenet_cifar.h5")
    weights = model.get_weights()
    print(f" size of 42 layer: {resize_to_2D(weights[42]).nbytes/2**20}")
    print(f" size of 42 layer: {weights[42].nbytes/2**20}")
    print(f" Back to 4d shape size :{resize_to_4D(resize_to_2D(weights[42]),weights[42]).nbytes/2**20}")
    size_of_sn = total_size(weights)
    layer_index_size = []
    for i in range(len(weights)):
       layer_index_size.append((i,np.shape(weights[i]),weights[i].nbytes/2**20))
    layer_index_size.sort(key=operator.itemgetter(2),reverse =True)
    # print(layer_index_size[0:10])
    print(f"Total size of the weight matrices:{size_of_sn}")
    # model.summary()

