import numpy as np
import tensorflow as tf
from data_sn import cifar_data
from conv_conversions import resize_to_2D

if __name__ == "__main__":

    model = tf.keras.models.load_model("./Important Squeezenets/sn_42_48_compression_test_15x15.h5")

    # x_train,y_train,x_test,y_test = cifar_data()
    # loss,acc= model.evaluate(x_test,y_test)
    # print(f"Loss:{loss}, Accuracy:{acc}")

    check_blocks_list=[(4,18)]

    weights = model.get_weights()

    block_size = (15,15)
    current_weight = resize_to_2D(weights[48])

    for r in check_blocks_list:
        # print(current_weight[r[0]*block_size[0]:(r[0]+1)*block_size[0],r[1]*block_size[1]:(r[1]+1)*block_size[1]])
        print(current_weight[r[0]*block_size[0]:r[0]*block_size[0]+5,r[1]*block_size[1]:r[1]*block_size[1]+5])


