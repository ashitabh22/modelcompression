import numpy as np
import math
import tensorflow as tf
from split_arr import convert



# Note: Weights_fc and Weights_conv are almost the same classes, Weights_fc was
# written much before and hence has some shortcomings but since its tested quite
# a bit. I have not changed it, if needed i can change it and combine both conv
# and fc into one.

# Known shortcomings of Weights_fc:
#     1) It cannot take rectangular block sizes,Weights_conv can.
#     2) It has a fixed criteria for formation of object, Weights_conv takes
#     checking_function which user can define and have more intricate object
#     creation criteria's
#     3)BLOCK_SIZE is a Class variable so dynamic blocksizes is not an option.
#        but that is not in option in Wiehgts_conv as well. Some changes will
#        need to be made for that to be possible.

def resize_to_3D(my_array,filter_width,filter_depth,num_filters):
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


def listof_validlayers(weights,layer_dimension):

    a = []
    layer_num =0
    for layer in weights:
        if len(np.shape(layer)) == layer_dimension:
            a.append(layer_num)
        layer_num +=1
    return a


def setup_weights_forcompression(model,orderbymetric= True,metric_for_orderby = None, absolute = None,only_some_layers = False,which_layers = None
                           , checking_function = None,args_dict = {},layer_type="fc"):
    #layer_type: "fc" or "conv"
    # Returns a dictionary of Weight_conv objects(IF orderbymetric is True)
    # which have been setup to use
    # orderby_centraltendency_conversion function.
    #
    if layer_type == "conv":
        if checking_function == None:
            print("No checking function passed,a checking function is compulsary \n \
                for the formation of Weight_cov objects\n")
            return
        weights = model.get_weights()
        if only_some_layers == True:
            valid_conv_index = which_layers
        else:
            valid_conv_index = listof_validlayers(weights,layer_dimension=3)
        weight_object_dict = {}
        active_layers = []
        for layer_index in valid_conv_index:
            # In the case of convolution layers, please note that Weights_conv
            # takes only 2d matrices, so they need to be converted to 2D and in the
            # end back to 3D
            temp_weights = resize_to_2D(weights[layer_index])
            temp = Weights_conv(temp_weights,checking_function = checking_function,args_dict = args_dict)
            if temp == None:
                pass
            else:
                if orderbymetric == True:
                    temp.set_ordered_coords(metric = metric_for_orderby,absolute = absolute)
                weight_object_dict[f'{layer_index}'] = temp
                active_layers.append(layer_index)
        return weight_object_dict,weights,active_layers

    if layer_type == "fc":
        weights = model.get_weights()
        valid_fc_index = listof_validlayers(weights,2)

        weight_object_dict = {}
        active_layers =[]
        for layer_index in valid_fc_index:
            temp = Weights_fc(weights[layer_index])

            if temp == None:
                pass
            else:
                if orderbymetric == True:
                    temp.set_ordered_coords(metric = metric_for_orderby,absolute = absolute)
                weight_object_dict[f'{layer_index}'] = temp
                active_layers.append(layer_index)
        return weight_object_dict,weights,active_layers


def check_on_blocksize(weight_matrix,args_dict):
    #Expecting a dict {'Block_dim':(rows,cols)}
    matrix_dim = np.shape(weight_matrix)
    block_dim = args_dict['Block_dim']
    if block_dim[0]>matrix_dim[0] or block_dim[1]>matrix_dim[1]:
        return False
    else:
        return True
class Weights_fc:

    NUM_OF_WEIGHTS =0
    BLOCK_SIZE=470
    CENTRAL_TENDENCY = "win_mean"
    CUTTING_RATIO = 0.2
    PERCENTILE = 0
    def __new__(cls,weight_matrix,order_by_metric = "mean"):
        # logger.info(f"General Information: \n Block Size: {Weights.BLOCK_SIZE} \n \
                    # Central Tendecy: {Weights.CENTRAL_TENDENCY} \n Cutting Ratio(used for winsorised mean): {Weights.CUTTING_RATIO},\n \
                    # Percentile: {Weights.PERCENTILE}")
        if all(i >= Weights_fc.BLOCK_SIZE for i in np.shape(weight_matrix)):
            return super(Weights_fc,cls).__new__(cls)
        else:
            print(f'Object with shape  was not formed because it was too small for the given Block Size')
            # logging.info(f'Object with shape {i} was not formed because it was too small for the given Block Size')

            return None


    def __init__(self,weight_matrix,order_by_metric = "mean"):
        self.array = weight_matrix
        self.original_rows,self.original_cols = np.shape(self.array)
        Weights_fc.NUM_OF_WEIGHTS +=1
        # logger.info("############################ NEW MATRIX ###################################")
        # logger.info("Initial setup of the weight matrix")
        self.reset()


    def set_ordered_coords(self,metric = None,absolute = True):
        num_of_x_blocks= int(self.new_end_row/self.BLOCK_SIZE)
        num_of_y_blocks = int(self.new_end_cols/self.BLOCK_SIZE)

        self.all_coords = tuple((i,j) for i in range(num_of_x_blocks)
                                        for j in range(num_of_y_blocks))
        # self.all_coords  = create_indices_for_fullrandomisation()
        # print(order_by_metric)
        self.list_active_coords = list(self.all_coords)
        self.list_passive_coords = []
        self.coords_centralmetric = [i for i in zip(self.all_coords,
                                                self.append_centraltendency_tuple_coords(central_tendency = metric,absolute = absolute))]
        self.sortedCoords_withmetric = sorted(self.coords_centralmetric,key=self.takeSecond)
        self.sortedCoords = list(zip(*self.sortedCoords_withmetric))[0]
        self.list_sorted_active_coords = list(self.sortedCoords)
        print(f" Matrix : {np.shape(self.array)}, Length of active list: {len(self.list_sorted_active_coords)}")
        # print(f" Matrix : {np.shape(self.array)}, Length of active list: {len(self.list_sorted_active_coords)}")
        print(len(self.list_sorted_active_coords))
        self.list_sorted_passive_coords = []

    def reset(self):

        self.curr_starting_row_block =0
        self.curr_starting_col_block =0
        self.new_end_row = self.original_rows - self.original_rows%self.BLOCK_SIZE
        self.new_end_cols = self.original_cols - self.original_cols%self.BLOCK_SIZE
        self.curr_ending_row_block =self.new_end_row # Fix this initialisation
        self.curr_ending_col_block =self.new_end_cols # Fix this initialisation
        self.stop_converting = False
        self.cut ="vertical"
        self.which_block =0

        # logging.info("Calling reset()")
        # logger.info(f"Shape of the matrix:({self.original_rows},{self.original_cols}) \n \
                     # Edit-able Rows,Columns: ({self.new_end_row},{self.new_end_cols}) \n \
                     # ** Last few rows and cols are left to maintain square shape of each block \n \
                     # Current Starting Row : {self.curr_starting_row_block} \n \
                     # Current Starting Column : {self.curr_starting_col_block} \n \
                     # Current Ending Column : {self.curr_ending_col_block} \n \
                     # Current Ending Row : {self.curr_ending_row_block}\n \
                     # Current Cut Config: {self.cut} \n ")

    def get_list_of_blocks(self,ratio):
        count = 0
        num_of_elements = len(self.list_active_coords)
        random.shuffle(self.list_active_coords)
        curr_blocks = []
        while count < math.ceil((num_of_elements*ratio)):
            curr_blocks.append(self.list_active_coords.pop())
            count+=1
        return curr_blocks

    def append_centraltendency_tuple_coords(self,central_tendency= None ,absolute =True):
        if central_tendency == None:
            print("No, metric passed. Returning None")
            return None
        print(f"Inside append_centraltendency_tuple_coords, central_tendency: {central_tendency}")
        list_of_answer = []
        for curr_block in self.all_coords:
            if absolute == True:
                temp = np.absolute(self.array[curr_block[0]*self.BLOCK_SIZE:(curr_block[0]+1)*(self.BLOCK_SIZE),
                                    curr_block[1]*self.BLOCK_SIZE:(curr_block[1]+1)*(self.BLOCK_SIZE)])
            else:

                temp = self.array[curr_block[0]*self.BLOCK_SIZE:(curr_block[0]+1)*(self.BLOCK_SIZE),
                            curr_block[1]*self.BLOCK_SIZE:(curr_block[1]+1)*(self.BLOCK_SIZE)]
            if central_tendency =="mean":
                average = np.mean(temp)
                list_of_answer.append(average)
            if central_tendency == "sum":
                sum_ = np.sum(temp)
                list_of_answer.append(sum_)
        return list_of_answer

    # @staticmethod
    # Convert to static method when you get time
    def takeSecond(self,elem):
        return elem[1]

    def orderby_centraltendency_conversion(self,conversion_type ="BCM",ratio =0.5,delete_num =0,testing = False):

        num_of_length = len(self.list_sorted_active_coords)
        counter =0

        curr_list = []
        # dequed_list_sorted_active_coords =deque(self.list_sorted_active_coords)
        while counter< math.ceil(num_of_length*ratio):
            curr_list.append(self.list_sorted_active_coords.pop(0))
            counter +=1

        for curr_block in curr_list:
            if testing == True:
                self.array[curr_block[0]*self.BLOCK_SIZE:(curr_block[0]+1)*(self.BLOCK_SIZE),
                           curr_block[1]*self.BLOCK_SIZE:(curr_block[1]+1)*(self.BLOCK_SIZE)] \
                = create_matrix(num = delete_num,rows = self.BLOCK_SIZE,cols = self.BLOCK_SIZE)
            else:

                self.array[curr_block[0]*self.BLOCK_SIZE:(curr_block[0]+1)*(self.BLOCK_SIZE),
                        curr_block[1]*self.BLOCK_SIZE:(curr_block[1]+1)*(self.BLOCK_SIZE)]= \
                    convert(self.array[curr_block[0]*self.BLOCK_SIZE:(curr_block[0]+1)*(self.BLOCK_SIZE),
                        curr_block[1]*self.BLOCK_SIZE:(curr_block[1]+1)*(self.BLOCK_SIZE)],block_size=(self.BLOCK_SIZE,self.BLOCK_SIZE),
                            central_tendency=self.CENTRAL_TENDENCY,cutting_ratio=self.CUTTING_RATIO,percentile=self.PERCENTILE,
                            conversion_matrix = conversion_type)

        self.list_sorted_passive_coords = self.list_sorted_passive_coords + curr_list
        # logger.info(f"COMPRESSED BLOCKS: {self.list_sorted_passive_coords}")
        # logger.info(f" Matrix: {np.shape(self.array)}, length: {len(self.list_sorted_active_coords)}")
        # print(self.list_sorted_active_coords)
        if len(self.list_sorted_active_coords) == 0:
            return 1
        else:
            return 0

    def get_matrix(self):
        # logger.info("get_matrix() called")
        return self.array

    def get_state(self):
        print(f' stop_converting: {self.stop_converting}')
    def printhi(self):
        print(' hi i exist')




# a = np.random.randint(0,10,size =[23,14])
# print(a)

# weights1 = Weights_fc(a)
# counter =0
# status = 0

# while status ==0:
#     status = weights1.fullrandomised_bc_conversion()
#     counter +=1

# print(weights1.get_matrix())

# a = np.random.randint(-10,10,size=[9,9])
# a_obj = Weights_fc(a,order_by_metric="sum")
# print(a_obj.get_matrix())
# a_obj.orderby_centraltendency_bcm_conversion(testing = True)
# print(a_obj.get_matrix())
# a_obj.orderby_centraltendency_bcm_conversion(delete_num = 2,testing = True)
# print(a_obj.get_matrix())



class Weights_conv:
    # Important distinction from Weights class, this class will
    # take rectangular block sizes as well.

    # BLOCK_DIM =(5,6)
    CENTRAL_TENDENCY = "win_mean"
    CUTTING_RATIO = 0.2
    PERCENTILE =0.5

    def __new__(cls,weight_matrix,checking_function,
                args_dict,order_by_metric="mean"):
        #checking function should return TRUE if object needs to be formed
        # FALSE if not. checking_function ONLY takes the weight matrix as the
        #arguement and a dictionary of arguements.

        if checking_function(weight_matrix,args_dict):
            return super(Weights_conv,cls).__new__(cls)
        else:
            print("The checking_function returned false,\
                  so no object was formed\n")

    def __init__(self,weight_matrix,checking_function,args_dict,
                 order_by_metric = "mean"):
        # Again, args_dict should have a 'Block_dim':(rows,cols)
        self.array = weight_matrix
        self.BLOCK_DIM = args_dict['Block_dim']
        self.original_rows,self.original_cols = np.shape(self.array)
        logger_string ="--Put in logger--"
        print(f"{logger_string} #### NEW MATRX ###")
        print(f"{logger_string} Doing initial setup...\n")
        self.reset()
    def reset(self):
        self.curr_starting_row_block =0
        self.curr_starting_col_block =0
        self.new_end_row = self.original_rows - self.original_rows%self.BLOCK_DIM[0]
        self.new_end_cols = self.original_cols - self.original_cols%self.BLOCK_DIM[1]
        self.curr_ending_row_block =self.new_end_row
        self.curr_ending_col_block =self.new_end_cols
        self.stop_converting = False

        print("Calling reset()")
        print(f"Shape of the matrix:({self.original_rows},{self.original_cols}) \n \
                     Edit-able Rows,Columns: ({self.new_end_row},{self.new_end_cols}) \n \
                     ** Last few rows and cols are left to maintain square shape of each block \n \
                     Current Starting Row : {self.curr_starting_row_block} \n \
                     Current Starting Column : {self.curr_starting_col_block} \n \
                     Current Ending Column : {self.curr_ending_col_block} \n \
                     Current Ending Row : {self.curr_ending_row_block}\n \
                     ")
    def set_ordered_coords(self,metric = None,absolute = True):
        # You need to run this function, if you want to use the "IN-ORDER"
        # algorithm. This function basically sets up a sorted list of all the
        # blocks based on a metric (mean,sum etc.)
        num_of_x_blocks= int(self.new_end_row/self.BLOCK_DIM[0])
        num_of_y_blocks = int(self.new_end_cols/self.BLOCK_DIM[1])

        self.all_coords = tuple((i,j) for i in range(num_of_x_blocks)
                                        for j in range(num_of_y_blocks))
        self.list_active_coords = list(self.all_coords)
        self.list_passive_coords = []
        self.coords_centralmetric = [i for i in zip(self.all_coords,
                                                self.append_centraltendency_tuple_coords(central_tendency = metric,absolute = absolute))]
        self.sortedCoords_withmetric = sorted(self.coords_centralmetric,key=self.takeSecond)
        self.sortedCoords = list(zip(*self.sortedCoords_withmetric))[0]
        self.list_sorted_active_coords = list(self.sortedCoords)
        # print(f"Sorted active coords: {self.list_sorted_active_coords}")
        print(f" Matrix : {np.shape(self.array)}, Length of active list: {len(self.list_sorted_active_coords)}")
        # print(f" Matrix : {np.shape(self.array)}, Length of active list: {len(self.list_sorted_active_coords)}")
        print(len(self.list_sorted_active_coords))
        self.list_sorted_passive_coords = []
    def takeSecond(self,elem):
        return elem[1]


    def append_centraltendency_tuple_coords(self,central_tendency= None ,absolute =True):
        if central_tendency == None:
            print("No, metric passed. Returning None")
            return None
        print(f"Inside append_centraltendency_tuple_coords, central_tendency: {central_tendency}")
        list_of_answer = []
        for curr_block in self.all_coords:
            if absolute == True:
                temp = np.absolute(self.array[curr_block[0]*self.BLOCK_DIM[0]:(curr_block[0]+1)*(self.BLOCK_DIM[0]),
                                    curr_block[1]*self.BLOCK_DIM[1]:(curr_block[1]+1)*(self.BLOCK_DIM[1])])
            else:

                temp = self.array[curr_block[0]*self.BLOCK_DIM[0]:(curr_block[0]+1)*(self.BLOCK_DIM[0]),
                            curr_block[1]*self.BLOCK_DIM[1]:(curr_block[1]+1)*(self.BLOCK_DIM[1])]
            if central_tendency =="mean":
                average = np.mean(temp)
                list_of_answer.append(average)
            if central_tendency == "sum":
                sum_ = np.sum(temp)
                list_of_answer.append(sum_)
        return list_of_answer

    # @staticmethod
    # Convert to static method when you get time
    def orderby_centraltendency_conversion(self,conversion_type ="BCM",ratio =0.5,delete_num =0,testing = False):

        num_of_length = len(self.list_sorted_active_coords)
        counter =0

        curr_list = []
        # dequed_list_sorted_active_coords =deque(self.list_sorted_active_coords)
        while counter< math.ceil(num_of_length*ratio):
            curr_list.append(self.list_sorted_active_coords.pop(0))
            counter +=1

        for curr_block in curr_list:
            if testing == True:
                self.array[curr_block[0]*self.BLOCK_DIM[0]:(curr_block[0]+1)*(self.BLOCK_DIM[0]),
                           curr_block[1]*self.BLOCK_DIM[1]:(curr_block[1]+1)*(self.BLOCK_DIM[1])] \
                = create_matrix(num = delete_num,rows = self.BLOCK_DIM[0],cols = self.BLOCK_DIM[1])
            else:

                self.array[curr_block[0]*self.BLOCK_DIM[0]:(curr_block[0]+1)*(self.BLOCK_DIM[0]),
                        curr_block[1]*self.BLOCK_DIM[1]:(curr_block[1]+1)*(self.BLOCK_DIM[1])]= \
                    convert(self.array[curr_block[0]*self.BLOCK_DIM[0]:(curr_block[0]+1)*(self.BLOCK_DIM[0]),
                        curr_block[1]*self.BLOCK_DIM[1]:(curr_block[1]+1)*(self.BLOCK_DIM[1])],block_size=(self.BLOCK_DIM[0],self.BLOCK_DIM[1]),
                            central_tendency=self.CENTRAL_TENDENCY,cutting_ratio=self.CUTTING_RATIO,percentile=self.PERCENTILE,
                            conversion_matrix = conversion_type)

        self.list_sorted_passive_coords = self.list_sorted_passive_coords + curr_list
        # logger.info(f"COMPRESSED BLOCKS: {self.list_sorted_passive_coords}")
        # logger.info(f" Matrix: {np.shape(self.array)}, length: {len(self.list_sorted_active_coords)}")
        # print(self.list_sorted_active_coords)
        if len(self.list_sorted_active_coords) == 0:
            return 1
        else:
            return 0

def create_matrix(num,rows,cols):
    return np.zeros([rows,cols]) + num







if __name__ == "__main__":

    # matrix  =np.ones((9,23))
    # arg_dict = {'Block_dim': (3,7)}
    # check_on_blocksize(matrix,arg_dict)
    # # Weights_conv takes weight matrix, a function, and its arguements
    # the function returns True or false based on whether the weight matrix
    #should be allowed to make weight_conve object or not.
    # obj_1 = Weights_conv(matrix,check_on_blocksize,args_dict = arg_dict)
    # obj_1.set_ordered_coords(metric = "mean")
    # obj_1.orderby_centraltendency_conversion(conversion_type="Toeplitz",testing = True)
    # print(obj_1.array)



    model = tf.keras.models.load_model("./alexnet_usethis_mnist.h5")
    args_dict = {}
    args_dict['Block_dim'] =(5,10)
    weight_dict,weights,active_layers = setup_conv_for_orderby(model,metric_for_orderby = "mean",
                                                               absolute = True, only_some_layers = True,
                                                               checking_function = check_on_blocksize,
                                                               which_layers = [18,24],
                                                               args_dict = args_dict )

    #Check the results of this and convert this back to 4d, and insert it into the
    #model again. It would be ideal if you could run the model and see some accuracy change.
    weight_dict['18'].orderby_centraltendency_conversion(conversion_type = "Toeplitz")
    weight_dict['24'].orderby_centraltendency_conversion(conversion_type = "Toeplitz")




