import tensorflow as tf
import numpy as np
from data_sn import cifar_data
import operator

def total_size(list_of_arrays):
    total =0
    for r in list_of_arrays:
        total = total + r.nbytes
    return total/2**20
if __name__ == "__main__":
    model =tf.keras.models.load_model("./squeezenet_cifar.h5")
    weights = model.get_weights()
    size_of_sn = total_size(weights)
    layer_index_size = []
    for i in range(len(weights)):
       layer_index_size.append((i,np.shape(weights[i]),weights[i].nbytes/2**20))
    layer_index_size.sort(key=operator.itemgetter(2),reverse =True)
    print(layer_index_size[0:5])
    print(f"Total size of the weight matrices:{size_of_sn}")
    # model.summary()

