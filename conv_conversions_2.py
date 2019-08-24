##modfied version
import numpy as np
def resize_to_4D(my_array,my_weights):
    num_filters = np.shape(my_weights)[3]
    filter_depth = np.shape(my_weights)[2]
    filter_width = np.shape(my_weights)[0]
    ctrx=0;
    ctry=0;
    my_conv=np.zeros([num_filters,filter_depth,filter_width,filter_width])
    for j in range(0,np.shape(my_array)[1],filter_width):
        for i in range(0,np.shape(my_array)[0],filter_depth):
            my_conv[ctry,:,:,ctrx]=my_array[i:i+filter_depth,j:j+filter_width]
            ctrx=ctrx+1
        ctrx=0
        ctry=ctry+1
    converted_conv=my_conv.transpose()
    return converted_conv


def resize_to_2D(my_weights):
    num_filters = np.shape(my_weights)[3]
    filter_depth = np.shape(my_weights)[2]
    filter_width = np.shape(my_weights)[0]
    x= filter_width*filter_depth
    y=filter_width * num_filters
    ctrx=0
    ctry=0
    temp_copy=np.zeros([x,y])
    trans=my_weights.transpose()
    for j in range(0,y,filter_width):
        for i in range(0,x,filter_depth):
            temp_copy[i:i+filter_depth,j:j+filter_width]=trans[ctry,:,:,ctrx]
            ctrx=ctrx+1
        ctrx=0
        ctry=ctry+1
    return temp_copy
