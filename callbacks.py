import tensorflow as tf
from data_sn import cifar_data
from conv_conversions import resize_to_4D,resize_to_2D
import logging
from weights_classes import Weights_fc,Weights_conv,check_on_blocksize,setup_weights_forcompression
from data import get_data_mnist,get_data_cifar
import numpy as np
import os

ResultSummary_CompressionStats = logging.getLogger(__name__)
ResultSummary_CompressionStats.setLevel(logging.INFO)
file_handler = logging.FileHandler('ResultSummary_CompressionStats.log')
formatter =logging.Formatter('%(process)d-%(levelname)s-%(asctime)s-%(message)s')
file_handler.setFormatter(formatter)
ResultSummary_CompressionStats.addHandler(file_handler)
random_key = np.random.randint(0,1000000)
experiment_key = f"squeezenet_cifar10_{random_key}"

folder_name = f"{experiment_key}"
os.mkdir(folder_name)
os.mkdir(f"./Experiments/{folder_name}/models")
file_handler_2= logging.FileHandler(f'./{folder_name}/ResultSummary_CompressionStats.log')
formatter =logging.Formatter('%(process)d-%(levelname)s-%(asctime)s-%(message)s')
file_handler_2.setFormatter(formatter)
ResultSummary_CompressionStats.addHandler(file_handler_2)

class MyLogger_Class():
    def __init__(self,heading,experiment_key):
        self.heading = heading
        self.experiment_key = experiment_key


class OrderBy_Callback(tf.keras.callbacks.Callback):

    def __init__(self,test_images,test_labels,leniency =0.1,types_of_layers=["fc"]):

        super().__init__()
        self.types_of_layers = types_of_layers # List of types of layers to compress ["fc"],["fc","conv"]
        ResultSummary_CompressionStats.info(" USING ORDERBY_CALLBACK ")
        from weights_classes import Weights_fc,Weights_conv,check_on_blocksize,setup_weights_forcompression
        self.test_images = test_images
        self.test_labels = test_labels
        # test_acc = .9936
        for layer_type in self.types_of_layers:

            if layer_type=="fc":
                self.metric = "mean"
                self.orderbymetric = True
                # leniency = 0.1
                # curr_test_acc = 100
                self.number_of_compression =1
                self.conversion_type = "Toeplitz"
                self.ratio =0.5
                self.absolute = True
                ResultSummary_CompressionStats.info(f" INITIAL STATE OF PARAMETERS: Block Size: {Weights_fc.BLOCK_SIZE} \n Order : {self.orderbymetric}  Order by metric : {self.metric} \n Leniency: {leniency} \n Conversion Type: {self.conversion_type} \n Stepping_Ratio = {self.ratio } \n Absolute: {absolute} \n ")
                self.weight_dict,self.weights,self.list_of_valid_index = setup_weights_forcompression(
                                                    model,orderbymetric = self.orderbymetric,metric_for_orderby = "mean",
                                            absolute = self.absolute,only_some_layers = False,layer_type = "fc")
                self.test_loss,self.test_acc = model.evaluate(self.test_images,self.test_labels)
                self.leniency = leniency
                # self.test_acc = 0.9936
                self.curr_test_acc = 100
                while (self.test_acc - self.leniency*self.test_acc)< self.curr_test_acc:
                    ResultSummary_CompressionStats.info(f"COMPRESSION NUMBER: {self.number_of_compression}")
                    for key,value in self.weight_dict.items():
                        ResultSummary_CompressionStats.info(f'****** Compressing {key} *********')
                        print(f'****** Compressing {key} *********')
                        ResultSummary_CompressionStats.info(f"Using Conversion Type: {self.conversion_type} ,Order By Central Tendency:{self.metric} Conversion with ratio: {self.ratio} ")
                        value.orderby_centraltendency_conversion(conversion_type = self.conversion_type,ratio = self.ratio )
                        # ResultSummary_CompressionStats.info(f"Number of uncompressed blocks: {value.list_sorted_active_coords}, Number of compressed blocks: {value.list_sorted_passive_coords}")
                        ResultSummary_CompressionStats.info(f"Number of uncompressed blocks: {len(value.list_sorted_active_coords)}, Number of compressed blocks: {len(value.list_sorted_passive_coords)}")
                        ResultSummary_CompressionStats.info(f"Uncompressed blocks: {value.list_sorted_active_coords[:50]}, Compressed blocks: {value.list_sorted_passive_coords[:50]}")
                    model.set_weights(self.weights)
                    self.number_of_compression +=1
                    self.curr_test_loss,self.curr_test_acc = model.evaluate(self.test_images,self.test_labels)
                    ResultSummary_CompressionStats.info(f"Loss: {self.curr_test_loss},Accuracy= {self.curr_test_acc}")
            if layer_type == "conv":
                self.metric_conv = "mean"
                self.orderbymetric_conv = True
                # leniency = 0.1
                # curr_test_acc = 100
                self.number_of_compression_conv =1
                self.conversion_type_conv = "Toeplitz"
                self.ratio_conv =0.5
                self.absolute_conv = True
                self.leniency_conv = leniency
                self.which_layers_conv =[30,42,48]
                self.args_dict = {'Block_dim':(15,15)}

                self.weight_dict_conv,self.weights_conv,self.list_of_valid_index_conv = setup_weights_forcompression(
                                                    model,orderbymetric = self.orderbymetric_conv,metric_for_orderby =self.metric_conv,
                                            absolute = self.absolute_conv,only_some_layers = True,which_layers=self.which_layers_conv,layer_type =layer_type,
                                            checking_function=check_on_blocksize,args_dict=self.args_dict)
                ResultSummary_CompressionStats.info(f" CONVOLUTION--INITIAL STATE OF PARAMETERS: Block Size: {self.args_dict['Block_dim']} \n Order : {self.orderbymetric_conv}  Order by metric : {self.metric_conv} \n Leniency: {self.leniency_conv} \n Conversion Type: {self.conversion_type_conv} \n Stepping_Ratio = {self.ratio_conv } \n Absolute: {self.absolute_conv} \n ")
                self.test_loss,self.test_acc = model.evaluate(self.test_images,self.test_labels)
                # self.test_acc = 0.9936
                self.curr_test_acc = 100 # Uncomment if fc is not run first.
                while (self.test_acc - self.leniency_conv*self.test_acc)< self.curr_test_acc:
                    ResultSummary_CompressionStats.info(f"CONVOLUTION --COMPRESSION NUMBER: {self.number_of_compression_conv}")
                    for key,value in self.weight_dict_conv.items():
                        ResultSummary_CompressionStats.info(f'****** Compressing {key} *********')
                        print(f'****** Compressing {key} *********')
                        ResultSummary_CompressionStats.info(f"CONVOLUTION-- Using Conversion Type: {self.conversion_type_conv} ,Order By Central Tendency:{self.metric_conv} Conversion with ratio: {self.ratio_conv} ")
                        value.orderby_centraltendency_conversion(conversion_type = self.conversion_type_conv,ratio = self.ratio_conv )
                        # This is an extra line which is different from FC's
                        self.weights_conv[int(key)]= resize_to_4D(value.get_matrix(),self.weights_conv[int(key)])
                        # ResultSummary_CompressionStats.info(f"Number of uncompressed blocks: {value.list_sorted_active_coords}, Number of compressed blocks: {value.list_sorted_passive_coords}")
                        ResultSummary_CompressionStats.info(f"CONVOLUTION--Number of uncompressed blocks: {len(value.list_sorted_active_coords)}, Number of compressed blocks: {len(value.list_sorted_passive_coords)}")
                        #Printing only first 50
                        ResultSummary_CompressionStats.info(f"CONVOLUTION--Uncompressed blocks: {value.list_sorted_active_coords[0:50]}, Compressed blocks: {value.list_sorted_passive_coords[:50]}")
                    # Setting the correct weights back into the model
                    # for layer in which_layers:
                    #     self.weights[layer] = self.weights_conv[layer]
                    # resize from 2d to 4d

                    model.set_weights(self.weights_conv)
                    self.number_of_compression_conv +=1
                    self.curr_test_loss,self.curr_test_acc = model.evaluate(self.test_images,self.test_labels)
                    ResultSummary_CompressionStats.info(f"CONVOLUTION--Loss: {self.curr_test_loss},Accuracy= {self.curr_test_acc}")
                    if self.number_of_compression_conv == 11:
                        print(" MODEL SAVED ################# ")
                        tf.keras.models.save_model(model,"sn_42_48_30_compression_test_15x15.h5")
                        break
                # print("CONVOLUTION--TESTING OVER")

    def on_train_batch_begin(self,batch,logs=None):
        # ResultSummary_CompressionStats.info("CHECK IF INSIDE")
        temp = model.get_weights()
        for layer_type in self.types_of_layers:
            if layer_type == "fc":
                for layer in self.list_of_valid_index:
                    all_compressed_chunks = self.weight_dict[f"{layer}"].list_sorted_passive_coords
                    len_ = len(all_compressed_chunks)
                    curr_block_size = Weights_fc.BLOCK_SIZE
                    # ResultSummary_CompressionStats.info(f"COMPRESSED BLOCKS, NUMBER OF COMPRESSED BLOCKS: {len_}")
                    # ResultSummary_CompressionStats.info(f"{all_compressed_chunks}")
                    for chunk in all_compressed_chunks:
                        temp[layer][chunk[0]*curr_block_size:(chunk[0]+1)*curr_block_size,
                                    chunk[1]*curr_block_size:(chunk[1]+1)*curr_block_size] = self.weight_dict[f"{layer}"].get_matrix()[chunk[0]*curr_block_size:(chunk[0]+1)*curr_block_size,
                                                                                                                        chunk[1]*curr_block_size:(chunk[1]+1)*curr_block_size]
            if layer_type =="conv":
                weights_temp = model.get_weights()
                for layer in self.list_of_valid_index_conv:
                    current_weights = weights_temp[layer]
                    current_weights_2d= resize_to_2D(current_weights)
                    all_compressed_chunks = self.weight_dict_conv[f"{layer}"].list_sorted_passive_coords
                    len_ = len(all_compressed_chunks)
                    # curr_block_size = Weights_fc.BLOCK_SIZE
                    block_dim = self.args_dict['Block_dim']
                    # ResultSummary_CompressionStats.info(f"COMPRESSED BLOCKS, NUMBER OF COMPRESSED BLOCKS: {len_}")
                    # ResultSummary_CompressionStats.info(f"{all_compressed_chunks}")
                    for chunk in all_compressed_chunks:
                        current_weights_2d[chunk[0]*block_dim[0]:(chunk[0]+1)*block_dim[0],
                                    chunk[1]*block_dim[1]:(chunk[1]+1)*block_dim[1]] = self.weight_dict_conv[f"{layer}"].get_matrix()[chunk[0]*block_dim[0]:(chunk[0]+1)*block_dim[0],
                                                                                                                          chunk[1]*block_dim[1]:(chunk[1]+1)*block_dim[1]]
                    weights_temp[layer] = resize_to_4D(current_weights_2d,weights_temp[layer])
                model.set_weights(weights_temp)

    def on_epoch_end(self,epoch,logs =None):
        ResultSummary_CompressionStats.info(f"EPOCH NUMBER: {epoch}")
        temp = model.get_weights()
        for layer_type in self.types_of_layers:
            if layer_type == "fc":
                for layer in self.list_of_valid_index:
                    all_compressed_chunks = self.weight_dict[f"{layer}"].list_sorted_passive_coords
                    len_ = len(all_compressed_chunks)
                    curr_block_size = Weights_fc.BLOCK_SIZE
                    # ResultSummary_CompressionStats.info(f"COMPRESSED BLOCKS, NUMBER OF COMPRESSED BLOCKS: {len_}")
                    # ResultSummary_CompressionStats.info(f"{all_compressed_chunks}")
                    for chunk in all_compressed_chunks:
                        temp[layer][chunk[0]*curr_block_size:(chunk[0]+1)*curr_block_size,
                                    chunk[1]*curr_block_size:(chunk[1]+1)*curr_block_size] = self.weight_dict[f"{layer}"].get_matrix()[chunk[0]*curr_block_size:(chunk[0]+1)*curr_block_size,
                                                                                                                        chunk[1]*curr_block_size:(chunk[1]+1)*curr_block_size]
            if layer_type =="conv":
                weights_temp = model.get_weights()
                for layer in self.list_of_valid_index_conv:
                    current_weights = weights_temp[layer]
                    current_weights_2d= resize_to_2D(current_weights)
                    all_compressed_chunks = self.weight_dict_conv[f"{layer}"].list_sorted_passive_coords
                    len_ = len(all_compressed_chunks)
                    # curr_block_size = Weights_fc.BLOCK_SIZE
                    block_dim = self.args_dict['Block_dim']
                    # ResultSummary_CompressionStats.info(f"COMPRESSED BLOCKS, NUMBER OF COMPRESSED BLOCKS: {len_}")
                    # ResultSummary_CompressionStats.info(f"{all_compressed_chunks}")
                    for chunk in all_compressed_chunks:
                        current_weights_2d[chunk[0]*block_dim[0]:(chunk[0]+1)*block_dim[0],
                                    chunk[1]*block_dim[1]:(chunk[1]+1)*block_dim[1]] = self.weight_dict_conv[f"{layer}"].get_matrix()[chunk[0]*block_dim[0]:(chunk[0]+1)*block_dim[0],
                                                                                                                          chunk[1]*block_dim[1]:(chunk[1]+1)*block_dim[1]]
                    weights_temp[layer] = resize_to_4D(current_weights_2d,weights_temp[layer])
                model.set_weights(weights_temp)
            # if layer_type =="conv":
            #     for layer in self.list_of_valid_index_conv:
            #         all_compressed_chunks = self.weight_dict_conv[f"{layer}"].list_sorted_passive_coords
            #         len_ = len(all_compressed_chunks)
            #         # curr_block_size = Weights_fc.BLOCK_SIZE
            #         block_dim = self.args_dict['Block_dim']
            #         # ResultSummary_CompressionStats.info(f"COMPRESSED BLOCKS, NUMBER OF COMPRESSED BLOCKS: {len_}")
            #         # ResultSummary_CompressionStats.info(f"{all_compressed_chunks}")
            #         for chunk in all_compressed_chunks:
            #             temp[layer][chunk[0]*block_dim[0]:(chunk[0]+1)*block_dim[0],
            #                         chunk[1]*block_dim[1]:(chunk[1]+1)*block_dim[1]] = self.weight_dict_conv[f"{layer}"].get_matrix()[chunk[0]*block_dim[0]:(chunk[0]+1)*block_dim[0],
            #                                                                                                             chunk[1]*block_dim[1]:(chunk[1]+1)*block_dim[1]]

            # model.set_weights(temp)
        # for layer in self.list_of_valid_index:
        #     all_compressed_chunks = self.weight_dict[f"{layer}"].list_sorted_passive_coords
        #     len_ = len(all_compressed_chunks)
        #     curr_block_size = Weights_fc.BLOCK_SIZE
        #     # ResultSummary_CompressionStats.info(f"COMPRESSED BLOCKS, NUMBER OF COMPRESSED BLOCKS: {len_}")
        #     # ResultSummary_CompressionStats.info(f"{all_compressed_chunks}")
        #     for chunk in all_compressed_chunks:
        #         temp[layer][chunk[0]*curr_block_size:(chunk[0]+1)*curr_block_size,
        #                     chunk[1]*curr_block_size:(chunk[1]+1)*curr_block_size] = self.weight_dict[f"{layer}"].get_matrix()[chunk[0]*curr_block_size:(chunk[0]+1)*curr_block_size,
        #                                                                                                           chunk[1]*curr_block_size:(chunk[1]+1)*curr_block_size]
        # model.set_weights(temp)
        # print(f"CHECKING IF CODE WORKING: {temp[36][3*470:3*470+4,7*470:7*470 +4]}")
        self.curr_test_loss,self.curr_test_acc = model.evaluate(self.test_images,self.test_labels)


        # print("CONVOLUTION TEST OVER")



        ResultSummary_CompressionStats.info(f" #### REQUIRED ACCURACY: {self.test_acc - self.leniency_conv*self.test_acc}")
        ResultSummary_CompressionStats.info(f" Current Test Accuracy in epoch {epoch}: {self.curr_test_acc}")
        for layer_type in self.types_of_layers:
            if layer_type=="fc":
                status =[ len(value.list_sorted_active_coords)==0 for key,value in self.weight_dict.items()]
                if all(status) == True:
                    ResultSummary_CompressionStats.info(f" All layers are fully compressed...")
                # if len(self.weight_dict["30"].list_sorted_active_coords) ==0 and len(self.weight_dict["36"].list_sorted_active_coords) ==0:
                #     ResultSummary_CompressionStats.info(f" Both layers are fully compressed...")
                    # model.stop_training = True
                else:
                    while (self.test_acc - self.leniency*self.test_acc)< self.curr_test_acc:
                        ResultSummary_CompressionStats.info(f"COMPRESSION NUMBER: {self.number_of_compression}")
                        for key,value in self.weight_dict.items():
                            ResultSummary_CompressionStats.info(f'****** Compressing {key} *********')
                            if len(value.list_sorted_active_coords) == 0:
                                ResultSummary_CompressionStats.info(f"Layer {key} is fully compressed...")
                            else:
                                print(f'****** Compressing {key} *********')
                                ResultSummary_CompressionStats.info(f'****** End of epoch: {epoch} Compressing {key} *********')
                                ResultSummary_CompressionStats.info(f"Using Conversion Type: {self.conversion_type} ,Order By Central Tendency:{self.metric} Conversion with ratio: {self.ratio}")
                                value.orderby_centraltendency_conversion(conversion_type = self.conversion_type,ratio = self.ratio )
                                ResultSummary_CompressionStats.info(f"Number of uncompressed blocks: {len(value.list_sorted_active_coords)}, Number of compressed blocks: {len(value.list_sorted_passive_coords)}")
                                ResultSummary_CompressionStats.info(f"Uncompressed blocks: {value.list_sorted_active_coords}, Compressed blocks: {value.list_sorted_passive_coords}")
                        model.set_weights(self.weights)
                        self.number_of_compression+=1
                        self.curr_test_loss,self.curr_test_acc = model.evaluate(self.test_images,self.test_labels)
                        ResultSummary_CompressionStats.info(f"Loss: {self.curr_test_loss},Accuracy= {self.curr_test_acc}")
                        status =[ len(value.list_sorted_active_coords)==0 for key,value in self.weight_dict.items()]
                        if all(status) == True:
                            ResultSummary_CompressionStats.info(f" All  layers are fully compressed...")
                            break
            if layer_type == "conv":

                status_conv= [ len(value.list_sorted_active_coords)==0 for key,value in self.weight_dict_conv.items()]
                if all(status_conv) == True:
                    ResultSummary_CompressionStats.info(" All Convolution layers are fully compressed ")
                else:

                    while (self.test_acc - self.leniency_conv*self.test_acc)< self.curr_test_acc:
                        ResultSummary_CompressionStats.info(f"CONVOLUTION --COMPRESSION NUMBER: {self.number_of_compression_conv}")
                        for key,value in self.weight_dict_conv.items():
                            ResultSummary_CompressionStats.info(f'****** Compressing {key} *********')
                            print(f'****** Compressing {key} *********')
                            ResultSummary_CompressionStats.info(f"CONVOLUTION-- Using Conversion Type: {self.conversion_type_conv} ,Order By Central Tendency:{self.metric_conv} Conversion with ratio: {self.ratio_conv} ")
                            value.orderby_centraltendency_conversion(conversion_type = self.conversion_type_conv,ratio = self.ratio_conv )
                            # This is an extra line which is different from FC's
                            self.weights_conv[int(key)]= resize_to_4D(value.get_matrix(),self.weights_conv[int(key)])
                            # ResultSummary_CompressionStats.info(f"Number of uncompressed blocks: {value.list_sorted_active_coords}, Number of compressed blocks: {value.list_sorted_passive_coords}")
                            ResultSummary_CompressionStats.info(f"CONVOLUTION--Number of uncompressed blocks: {len(value.list_sorted_active_coords)}, Number of compressed blocks: {len(value.list_sorted_passive_coords)}")
                            # ResultSummary_CompressionStats.info(f"CONVOLUTION--Uncompressed blocks: {value.list_sorted_active_coords}, Compressed blocks: {value.list_sorted_passive_coords}")
                        # Setting the correct weights back into the model
                        # for layer in which_layers:
                        #     self.weights[layer] = self.weights_conv[layer]
                        # resize from 2d to 4d

                        model.set_weights(self.weights_conv)
                        self.number_of_compression_conv +=1
                        self.curr_test_loss,self.curr_test_acc = model.evaluate(self.test_images,self.test_labels)
                        ResultSummary_CompressionStats.info(f"CONVOLUTION--Loss: {self.curr_test_loss},Accuracy= {self.curr_test_acc}")
                    # if len(self.weight_dict["30"].list_sorted_active_coords )==0 and len(self.weight_dict["36"].list_sorted_active_coords) ==0:
                    #     ResultSummary_CompressionStats.info(f" Both layers are fully compressed...")
                        # break
        tf.keras.models.save_model(model,f"./{folder_name}/nn_{epoch}.h5")










if __name__ == "__main__":

    # model = tf.keras.models.load_model('./base_models/alexnet_usethis_mnist.h5')





    mylogger = MyLogger_Class(heading = "######## SQUEEZENET  WALTing Compression\n",
                             experiment_key = experiment_key)
    ResultSummary_CompressionStats.info(mylogger.heading)
    ResultSummary_CompressionStats.info(mylogger.experiment_key)

    # train_images,train_labels,test_images,test_labels  = get_data_mnist()

    # # log the dataset numbers here, add later.
    # total_num = 10002
    # batch_size = 64
    # num_of_epochs =10

    # model.fit(train_images[:total_num],train_labels[:total_num],batch_size=batch_size,
    #                        epochs = num_of_epochs,callbacks=[OrderBy_Callback(test_images = test_images,
    #                                                                           test_labels = test_labels,
    #                                                                           types_of_layers=["conv"])])

    # tf.keras.models.save_model(model,'modelname.h5')
    model = tf.keras.models.load_model('./squeezenet_cifar.h5')
    x_train,y_train,test_images,test_labels= cifar_data()

    model.fit(x_train,y_train,batch_size = 64,epochs =15,
              callbacks=[OrderBy_Callback(test_images = test_images,
                                          test_labels = test_labels,
                                          leniency=0.05,
                                          types_of_layers=["conv"])])








