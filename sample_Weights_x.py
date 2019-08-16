from weights_classes import Weights_fc,Weights_conv,check_on_blocksize,setup_weights_forcompression
import tensorflow as tf
import numpy as np


if __name__ == "__main__":

    # Using Weights_conv
    #Uncomment to run

    # matrix = np.random.random_integers(0, 100, (7,17))
    # arg_dict = {'Block_dim':(5,5)}
    # obj_1 = Weights_conv(matrix,check_on_blocksize,args_dict = arg_dict)
    # obj_1.set_ordered_coords(metric = "mean",absolute =True)
    # obj_1.orderby_centraltendency_conversion(conversion_type ="Toeplitz")
    # print(obj_1.list_sorted_passive_coords)
    # print( obj_1.array )


    ## Using Weights_fc
    #Uncomment to run

    #matrix = np.random.randint(0, 100, (7,17))
    ##[IMPORTANT]Need to change BLOCK_SIZE Class variable manually
    #obj_fc = Weights_fc(matrix,order_by_metric = "mean")
    #obj_fc.set_ordered_coords(metric = "mean",absolute = True)
    #obj_fc.orderby_centraltendency_conversion(conversion_type = "Toeplitz")
    #print(obj_fc.list_sorted_passive_coords)
    #print(obj_fc.array)


    # Using setup_weights_forcompression

    # model = tf.keras.models.load_model("./base_models/alexnet_usethis_mnist.h5")
    # weight_fc_dict, weights_fc,active_fc_layers = setup_weights_forcompression(
    #     model,orderbymetric = True,metric_for_orderby = "mean",
    #     absolute = True,only_some_layers = False,layer_type = "fc")

    # for key,value in weight_fc_dict.items():
    #     print(f"The layers that have been setup = {key}")
    # weight_fc_dict["30"].orderby_centraltendency_conversion(conversion_type = "Toeplitz")
    # weight_fc_dict["30"].get_matrix()



