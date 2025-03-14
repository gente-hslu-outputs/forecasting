
from keras import Model
from keras.layers import Input, Dense, Lambda, Concatenate, Cropping1D, Flatten, Conv1D, Add, SpatialDropout1D, Layer
import tensorflow as tf
import numpy as np

from logging import INFO,log
import logging

logging.basicConfig(level=INFO)
logger = logging.getLogger(__name__)




# TODO: Move into TCN
def tcn_residual_block(inputs:Layer,
                       filters=1,
                       kernel_size=2,
                       dilation_rate=None,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal',
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       use_bias=False,
                       dropout_rate=0.0,
                       layer_nr:int= 0,                 # layer nr of current tcn layer
                       data_stream_nr:int = 0           # which dimension of datastream?
                       ):
    """
    TCN Residual Block.
    TCN uses zero-padding to maintain `steps` value of the ouput equal to the one in the input.
    See [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling]
    (https://arxiv.org/abs/1803.01271).
    A Residual Block is obtained by stacking togeather (2x) the following:
        - 1D Dilated Convolution
        - WeightNorm (here absent)
        - ReLu
        - Spatial Dropout
    And adding the input after trasnforming it with a 1x1 Conv
    # Arguments
        intpus (layer): Previous Layer of Residual Block
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Usually dilation rate increases exponentially with the depth of the network.
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
    # Input shape
        3D tensor with shape: `(batch, steps, n_features)`
    # Output shape
        3D tensor with shape: `(batch, steps, filters)`
    """

    # Conv - Conv Block (with Dropout)
    outputs = Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     use_bias=use_bias,
                     bias_initializer=bias_initializer,
                     bias_regularizer=bias_regularizer,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     padding='causal',
                     dilation_rate=dilation_rate,
                     activation='relu',
                     name = f"Conv_1D_D{data_stream_nr}_L{layer_nr:02}_1")(inputs)
    outputs = SpatialDropout1D(dropout_rate, trainable=True)(outputs)                       # intermediate dropout layer
    
    outputs = Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     use_bias=use_bias,
                     bias_initializer=bias_initializer,
                     bias_regularizer=bias_regularizer,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     padding='causal',
                     dilation_rate=dilation_rate,
                     activation='relu',
                     name=f"Conv_1D_D{data_stream_nr}_L{layer_nr:02}_2"
                     )(outputs)
    
    outputs = SpatialDropout1D(dropout_rate, trainable=True)(outputs)
    
    # Residual Connection (with Kernel Size = 1)
    skip_out = Conv1D(filters=filters, kernel_size=1, activation='linear', name = f'Residual_{data_stream_nr}_{layer_nr:02}')(inputs)          # 1x1 Residual Block
    
    # Adding both back toghether
    residual_out = Add()([outputs, skip_out])
    
    # Wrap the output with a Lambda layer to assign ONE name to the block
    tcn_block = tf.keras.layers.Lambda(lambda x: x, name=f"TCN_{layer_nr}_{data_stream_nr}")(residual_out)

    return tcn_block




class TemporalConvolutionalNetwork():
    """
    Adaption TCN Network from Gasparin

    Create a TCN Model with Resiudual Block functionalities
    BUT dont only use 1 output node from network, but outwindow nodes
    """

    def __init__(self,
                 layers=None,  #
                 nr_of_data_streams = 1,  # dimension of input data (how many data sets)
                 filters=1,  #
                 kernel_size=2,  #
                 dilation_rate=2,  #
                 kernel_initializer='glorot_normal',
                 bias_initializer='glorot_normal',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 use_bias=False,
                 dropout_rate=0.0, # add dropout
                 return_sequence=False):                    
        '''
        :param layers: nr of TCN Layers
        :param filters: nr of pararel filters
        :param kernel_size: size of TCN kernel
        :param dilation_rate: dilation rate of kernel
        :param kernel_initializer: initializer type
        :param bias_initializer: initializer type
        :param kernel_regularizer: regularizer type
        :param bias_regularizer: regularizer type
        :param use_bias:    do you want to use the bias
        :param dropout_rate: i mean you should know what this does, if you read this
        '''

        self.layers = layers
        self.nr_of_data_streams = nr_of_data_streams
        self.dilation_rate = dilation_rate
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.return_sequence = return_sequence


        self.built = False

        # Calculate Receptive Field
        # From TCN Rempy (Picture: https://github.com/philipperemy/keras-tcn#receptive-field)
        self.receptive_field = np.sum([np.power(dilation_rate,hidden)*(kernel_size-1) for hidden in range(0,layers)])+1

        

    def build_model(self, input_window:int, output_window:int, nr_of_dense_layers = 1)-> Model:

        '''
        Create a Model that takes as inputs:
                     - 3D Tensor of shape (batch_size, input_window, input_dim_data) and outputs:
                     - 2D tensor of shape (batch_size, output_window)

            Parameters
                input_window (int):         input sequence length (in timesteps)
                output_window: (int):       output sequence length (in timesteps)
                nr_of_dense_layers (int):   How many Dense Layers (Backbone), to add after TCN, defaults  to 1
                
            Returns
                TCN Model (TF Model):       Built Tenserflow Model

        
        Example  TCN Architecture (with 3 Inputs)

            TCN 1       TCN 2       TCN 3
            |   |       |   |       |   |
            |   |       |   |       |   |
            |   |       |   |       |   |
            \                           /
             \      Concat Layer       /
              \-----------------------/
                    |    :   :    |
        |            Flatten Layer          |
         ------------ Dense 1 --------------
          \                                /
           \                              /
              --------- Dense 2 --------
               \                      /
                \                    /
                   ----- Output -----
        

        '''
        self.output_window = output_window
        self.input_winndow = input_window


        # inputs
        inputs = Input(shape=(input_window, self.nr_of_data_streams), name='inputs')        # shape [I_w * dimension_data]

        # Function to split the tensor
        def split_tensor(x):
            return tf.split(value=x, num_or_size_splits=self.nr_of_data_streams, axis=2)

        # Use Lambda layer to apply the splitting function
        input_splits = Lambda(split_tensor)(inputs)


        # # Split Tensor into List of Substacks (if nr_of_data_streams = 1 --> [inputs])
        # input_slits = tf.split(value = inputs,
        #                     num_or_size_splits= [1 for _ in range(self.nr_of_data_streams)],
        #                     axis=2) # splot along axis of nr_of dimensions
        
        logging.info(f"Created {len(input_splits)} Input Streams of Size {tf.shape(input_splits[0])}")

        # For each Data Stream, add a Paralell TCN Net, which gets merged (concat) after
        for j in range(self.nr_of_data_streams):

            for i in range(self.layers):
                # reset layer reference to input layer for each dimension in dataset
                if i == 0:
                    # connect to correct input tensor 
                    # maybe split into parts
                    layer = input_splits[j]

                layer = tcn_residual_block(inputs= layer,                       # 1st time connect tcn as (layer = inputs)
                                           filters=self.filters,
                                           kernel_size=self.kernel_size,
                                           use_bias=self.use_bias,
                                           bias_initializer=self.bias_initializer,
                                           bias_regularizer=self.bias_regularizer,
                                           kernel_initializer=self.kernel_initializer,
                                           kernel_regularizer=self.kernel_regularizer,
                                           dilation_rate=self.dilation_rate ** i,
                                           dropout_rate=self.dropout_rate,
                                           layer_nr=i,
                                           data_stream_nr = j)
            # get correct output length

            # Version TCN REMY (observed a bad performance --> switch to Huber Version)
            # c = inputs.shape[1] - 1                               # leave only ONE Value (causal case) -> TCN Remy
            # Version Patrick Huber #
            c = inputs.shape[1] - self.output_window                # leave output_window values
            out_layer_TCN = Cropping1D(cropping=(c,0))(layer)       # cut c samples from beginning of output "sequence"
            out_layers = []                                         # A merge layer should be called on a list of inputs
            out_layers.append(out_layer_TCN)                        # add layer to out layers (for each dimension)

        # Merge the Layers
        if len(out_layers) > 1:
            concat = Concatenate(axis=-1)(out_layers)
        else:
            concat = out_layers[0]
        
        # Added Flatten Layer to have 1D structure
        # With Dense, the Temporal Positioning does not matter anymore.
        flatten_layer = Flatten()(concat)

        # rename layer -> for easier building of dense stack
        previous_layer = flatten_layer
        # always add aone dense layer

        # ---------- Add Dense Backbone ----------------
        for layer_nr in range(nr_of_dense_layers-1):
            
            # make layer bigger, for deeper Dense nets to have this architecture:
            dense_layer = Dense(units=self.output_window * (nr_of_dense_layers - layer_nr),       
                                                activation='relu',
                                                kernel_initializer=self.kernel_initializer,
                                                bias_initializer=self.kernel_initializer)(previous_layer)
            
            previous_layer = dense_layer    # set previous layer to this dense layer

        # last dense layer
        output_layer = Dense(units=self.output_window,
                                                activation='linear',            # Needs to be linear -> can have negative outputs if normalized values are used
                                                kernel_initializer=self.kernel_initializer,
                                                bias_initializer='zeros')(previous_layer)

        # Connect model with in and outputs
        tcn_model:Model = Model(inputs=[inputs], outputs=output_layer)
        self.built = True  # set build = true
        
        return tcn_model

    # commented out -> no use
    # def call(self, inputs, training=True):
    #     if not self.built: raise ValueError("Model called before it has been built. Call first `build_model` on the model.")
    #     return self.model(inputs, training=training)         


# ----------------------------------------------------------------
# TODO: consider making part of Class (and __init__ method?)

def calculate_tcn_layers(input_window:int, dillation:int, kernel_size:int) -> tuple[int,int]:
    """ 
    Calculate the needed number of TCN layers given an input_window size and a Dillation rate
    Such that the last TCN Neuron sees the entire input_window

    Parameters:
        input_window (int):     How many time steps are in input sequence
        dillation (int):        How far do TCN kernels spreach apart each layer
        kernel_size (int):      How wide is a TCN Filter

    Returns:
        tcn_layers, receptive_field (Tuple[int,int]):   Tuple of how many layers to use and how big the receptive field is
    """
    tcn_layers:int = 2              # start at 2 layers ??
    receptive_field:int = 0

    # assume this is correct -> made by pascal jund
    while receptive_field < input_window:
        receptive_field = np.sum([np.power(dillation, hidden) * (kernel_size - 1) for hidden in range(0, tcn_layers)]) + 1
        tcn_layers += 1 
    
    return tcn_layers, receptive_field

# unused
def create_tcn_model(input_dim:int, input_window:int, output_window:int)->Model:
    """
    Creates a TCN Model based on some __init__ parameters and some of the parameters below
    
        Parameters:
            input_dim (int):            How many input data streams are there?
            input_window (int):         How many timesteps there are in the model input
            output_window (int):        How many timesteps there are in the model output
        Returns:
            tcn_model (Model):  Complete TCN Model including Dense Backbone

    """
    # Model HYPER Parameter TCN Net
    DILATION: int = 2                                               # dilation rate of tcn filter (default = 2)
    PARALLEL_FILTERS:int = 16                                       # nr of parallel filters of TCN network
    KERNEL_SIZE: int = int(np.min([input_window/6,4]))              # kernel size of tcn filter (not bigger than 4)
    DENSE_HIDDEN:int = 1                                            # nr of hidden dense layers (will be >=2)
    DROPOUT_RATE:float = 0.005
    
    # calculate nr of tcn layers needed for the current input window
    tcn_layers, receptive_field = __calculate_tcn_layers(input_window, DILATION, KERNEL_SIZE)
    
    log(INFO,f"selected {tcn_layers} tcn layers with receptive field of {receptive_field}")

    tcn = TemporalConvolutionalNetwork(layers=tcn_layers,   # how many layers
                    nr_of_data_streams=input_dim,           # how many data streams as input
                    filters=PARALLEL_FILTERS,
                    kernel_size=KERNEL_SIZE,  #
                    dilation_rate=DILATION,  #
                    kernel_initializer='glorot_normal',
                    bias_initializer='glorot_normal',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    use_bias=False,
                    dropout_rate=DROPOUT_RATE)  # add dropout)

    model = tcn.build_model(input_window=input_window,
                            output_window=output_window,
                            nr_of_dense_layers=DENSE_HIDDEN)
    return model
        
