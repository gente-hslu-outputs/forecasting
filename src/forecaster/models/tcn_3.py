"""
        DEPRECATED 
            TRIAL FOR TF 2.17 -> does not work

"""


import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Concatenate, Cropping1D, Flatten, Conv1D, Add, SpatialDropout1D, Layer
from keras import Model
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the TCN Block as a Keras Layer
class TCN(Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.0, use_bias=False, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        # Initialize layers as None; they will be created in build()
        self.conv1 = None
        self.dropout1 = None
        self.conv2 = None
        self.dropout2 = None
        self.residual_conv = None

    def build(self, input_shape):
        self.conv1 = Conv1D(filters=self.filters,
                            kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate,
                            padding='causal',
                            use_bias=self.use_bias,
                            activation='relu')
        
        self.dropout1 = SpatialDropout1D(self.dropout_rate)
        
        self.conv2 = Conv1D(filters=self.filters,
                            kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate,
                            padding='causal',
                            use_bias=self.use_bias,
                            activation='relu')
        
        self.dropout2 = SpatialDropout1D(self.dropout_rate)
        
        # Residual connection to match the input and output shapes
        self.residual_conv = Conv1D(filters=self.filters, kernel_size=1, activation='linear')

        # Mark the layer as built
        super(TCN, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        residual = self.residual_conv(inputs)
        return Add()([x, residual])

    def compute_output_shape(self, input_shape):
        return input_shape

# Define a Custom Layer to Handle TensorFlow Operations
class SplitTensor(Layer):
    def __init__(self, num_or_size_splits, axis=2, **kwargs):
        super(SplitTensor, self).__init__(**kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def build(self, input_shape):
        # This layer doesn't have any trainable weights or configurations that depend on input_shape,
        # but we'll still implement build() to follow best practices.
        # Calling the superclass build method marks the layer as built.
        super(SplitTensor, self).build(input_shape)

    def call(self, inputs):
        return tf.split(inputs, num_or_size_splits=self.num_or_size_splits, axis=self.axis)

    def compute_output_shape(self, input_shape):
        # Calculate the shape of the output tensor(s) based on the split
        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] // self.num_or_size_splits
        return [tuple(output_shape) for _ in range(self.num_or_size_splits)]

    
# Define the TCN_Mode class
class TCN_Builder:
    def __init__(self, layers, nr_of_data_streams, filters, kernel_size, dilation_rate, dropout_rate=0.0):
        self.layers = layers
        self.nr_of_data_streams = nr_of_data_streams
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        # From TCN Rempy (Picture: https://github.com/philipperemy/keras-tcn#receptive-field)
        self.receptive_field = np.sum([np.power(dilation_rate,hidden)*(kernel_size-1) for hidden in range(0,layers)])+1

    def build_model(self, input_window, output_window, nr_of_dense_layers=1):
        self.input_window = input_window
        self.output_window = output_window
        
        inputs = Input(shape=(input_window,self.nr_of_data_streams), name='Input')
        
        # Use the custom SplitTensor layer to handle the tf.split operation
        input_splits = SplitTensor(num_or_size_splits=self.nr_of_data_streams, name = f"Slit_{self.nr_of_data_streams}D")(inputs)
        out_layers = []
        
        # create j TCN Stacks (1 for each Data Stream)
        for j in range(self.nr_of_data_streams):
            x = input_splits[j]     # each data stream starts with its split tensor
            for i in range(self.layers):
                # create a Stack of TCN Layers for each data stream
                x = TCN(filters=self.filters,
                        kernel_size=self.kernel_size,
                        dilation_rate=self.dilation_rate ** i,
                        dropout_rate=self.dropout_rate,
                        name = f"TCN_L{i}_D{j}")(x)
            
            # Add all TCN Stacks together    
            out_layers.append(x)

        # Add all Columns (Dimensions Togheter)
        added = Add(name=f"Add_L{i}")(out_layers)
      
        
        # # Crop Away any timesteps whic
        # c = input_window - output_window
        # cropped_output = Cropping1D(cropping=(c, 0), name="Cropping")(added)
        
        flatten_layer = Flatten()(added)
        previous_layer = flatten_layer
        
        # Add Hidden Dense Layers
        for layer_nr in range(nr_of_dense_layers - 1):
            
            # make them smaller and smaller -> to condense layers to output_window (if multiple)
            dense_layer = Dense(units=output_window * (nr_of_dense_layers - layer_nr),
                                activation='relu',
                                name = f"Hidden_Dense_{layer_nr}")(previous_layer)
            previous_layer = dense_layer
        
        output_layer = Dense(units=output_window,
                             activation='linear',
                             name = "Output")(previous_layer)
        tcn_model = Model(inputs=[inputs], outputs=output_layer)
        return tcn_model


def create_simple_tcn(input_window:int,input_dim:int,output_window:int)->Model:
    """ 
    Create a TCN Model

    Parameters:
        input_window (int):     Nr of Timesteps
        input_dim (int):        Nr of Features
        output_window (int):    Nr of Timesteps
    
    Returns:
        model (Model):      Keras Model
    """
    
    
    # init a tcn builder 
    tcn = TCN_Builder(layers = 6,
                      nr_of_data_streams=input_dim,
                      filters=10,
                      kernel_size=3,
                      dilation_rate=2,
                      dropout_rate=0.001)
    
    # build the model 
    model = tcn.build_model(input_window=input_window,
                            output_window=output_window,
                            nr_of_dense_layers=1)
    
    logging.info(f"Created TCN Model with receptive field of {tcn.receptive_field} for input window of {tcn.input_window}")

    return model

if __name__ == "__main__":
    # Example usage
    input_window = 4 * 24
    output_window =  4 * 24
    nr_of_data_streams = 3
    
    model = create_simple_tcn(input_window=input_window,
                              input_dim=nr_of_data_streams,
                              output_window=output_window)
    
    model.summary()
