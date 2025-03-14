import logging

import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten, Input, Conv2D, Conv1D, MaxPool1D, MaxPool2D

# DEPRECATED -> use tcn_2.py instead
# Local import
# from forecaster.models.temporal_convolutional_network import TemporalConvolutionalNetwork, calculate_tcn_layers
# from forecaster.models import TemporalConvolutionalNetwork, calculate_tcn_layers

# DEPRECATED -> use tcn_2.py instead
# def create_simple_tcn(input_window:int,input_dim:int,output_window:int)->Model:
#     """ 
#     Create a TCN Model

#     Parameters:
#         input_window (int):     Nr of Timesteps
#         input_dim (int):        Nr of Features
#         output_window (int):    Nr of Timesteps
    
#     Returns:
#         model (Model):      Keras Model
#     """
#     logging.info(f"Creating TCN Model")

#     # calculate nr of tcn layers needed for the current input window
#     tcn_layers, receptive_field = calculate_tcn_layers(input_window, dillation=2, kernel_size=2)
    
#     logging.info(f"selected {tcn_layers} tcn layers with receptive field of {receptive_field}")

#     tcn = TemporalConvolutionalNetwork(layers=tcn_layers,
#                                        nr_of_data_streams=input_dim,
#                                        filters=12)
    
#     model = tcn.build_model(input_window=input_window,
#                             output_window=output_window,
#                             nr_of_dense_layers=1)
    
#     return model


def create_simple_cnn(input_window:int,input_dim:int,output_window:int)->Model:
    """
    Create a Simple CNN Model. Dynamically switches between Conv1D and Conv2D (depending on input_dim)

    Parameters:
        input_window (int):     Nr of Timesteps
        input_dim (int):        Nr of Features
        output_window (int):    Nr of Timesteps
    
    Returns:
        model (Model):      Keras Model
    """
    
    logging.info(f"Creating CNN Model")

    model = Sequential()

    # dynamically choose correct Layer Shapes
    # Conv = Conv2D
    # kernel_size = (np.max([3,input_dim]),input_dim)
    # MaxPool = MaxPool2D
    # pool_size = (input_dim,input_dim)
    # if input_dim == 1:
    
    Conv = Conv1D
    kernel_size = 3
    MaxPool = MaxPool1D
    pool_size = 2

    model.add(Input(shape=(input_window,input_dim)))

    nr_of_filters = input_dim * 5
    
    # calculate how many CONV CONV MaxPool Blocks you can do (and stay above 10% window length)
    kernel_window = input_window
    nr_of_blocks = 0

    while kernel_window > int(input_window/10):
        nr_of_blocks += 1
        kernel_window -= kernel_size
        kernel_window /= pool_size

    # go 1 step back
    nr_of_blocks -= 1

    for _ in range(nr_of_blocks):
        # CONV CONV MaxPool Block
        model.add(Conv(filters = nr_of_filters,kernel_size=kernel_size, activation="relu"))
        model.add(Conv(filters = nr_of_filters,kernel_size=kernel_size, activation="relu"))   
        model.add(MaxPool(pool_size=pool_size))

    # Feature Selection (Dense Backbone)
    model.add(Flatten())
    # model.add(Dense(units = input_window, activation="relu"))
    model.add(Dense(units=output_window,activation="linear"))

    return model

def create_simple_dense(input_window:int,input_dim:int,output_window:int)->Model:
    """
    Create a Simple Dense Model for Inputs of Shape (Input_Window, Input_Dim)

    Parameters:
        input_window (int):     Nr of Timesteps
        input_dim (int):        Nr of Features
        output_window (int):    Nr of Timesteps
    
    Returns:
        model (Model):      Keras Model
    """
    model = Sequential()
    model.add(Input(shape=(input_window,input_dim)))

    model.add(Flatten())
    for i in range(2):
        model.add(Dense(units = input_window, activation="relu"))

    model.add(Dense(units=output_window,activation="linear")) 

    return model


if __name__ == "__main__":

    model_pv = create_simple_cnn(input_window=24,input_dim=3,output_window=24)
    model_pv.summary()
