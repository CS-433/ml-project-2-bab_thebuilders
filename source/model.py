
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *

def squeeze_excite_block(inputs, ratio=8):
    ''' 
    Architectural unit designed to improve the representational power of a network by enabling it to perform dynamic channel-wise feature              recalibration. 
    Args: 
        inputs : convolutional block 
    Returns:
        x :     
    
    '''
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    # Each channel is "squeezed" into a single numeric value using average pooling.
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    
    # A dense layer followed by a ReLU adds non-linearity and output channel complexity is reduced by a ratio.
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    
    # Another dense layer followed by a sigmoid gives each channel a smooth gating function.
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    # Finally, we weight each feature map of the convolutional block based on the side network; the "excitation".
    x = Multiply()([init, se])
    return x

def conv_block(inputs, filters):
    '''
    Implements a convolution block.
    
    Args: 
        inputs : ?????
        filter: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution), filter to be applied during         the process  
    Returns:
        x : convolution block
    '''
    
    x = inputs

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x

def encoder1(inputs):
    '''
    
    Args:       
        inputs              : 
    Returns:
        output              :
        skip_connections    : 
    
    '''
    skip_connections = []

    model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]

    # keep in memory the outputs of the last layer of each block
    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("block5_conv4").output
    return output, skip_connections

def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]
    skip_connections.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f)

    return x

## lead for another pretrained model as encoder for the second Unet
# def encoder2(inputs):
#     skip_connections = []
#
#     output = DenseNet121(include_top=False, weights='imagenet')(inputs)
#     model = tf.keras.models.Model(inputs, output)
#
#     names = ["input_2", "conv1/relu", "pool2_conv", "pool3_conv"]
#     for name in names:
#         skip_connections.append(model.get_layer(name).output)
#     output = model.get_layer("pool4_conv").output
#
#     return output, skip_connections

def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)

    return x, skip_connections

def decoder2(inputs, skip_1, skip_2):
    num_filters = [256, 128, 64, 32]
    skip_2.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_1[i], skip_2[i]])
        x = conv_block(x, f)

    return x

def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x

def Upsample(tensor, size):
    """Bilinear upsampling"""
    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)
    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)

def ASPP(x, filter):
    '''
    Performs Atrous Spatial Pyramidal Pooling on input data, applying a specified filter.
    It follows the architecture developped by the DeepLabV3.
    
    Args:
        x: input data
        filter: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution), filter to be applied during         the process
    
    Returns:
        y : ouput data
    '''
    shape = x.shape

    
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def build_model(shape):
    '''
    Builds the model of the double U-Net with all the layers created for the task of road segmentation.
    The results of both U-Nets are going to be returned.
    
    Args:
        shape: tuple, representing the shape of the input on which the model should be built.
    Returns:
        model : tensorflow model
    '''
    
    inputs = Input(shape)
    
    # U-Net n°1
    x, skip_1 = encoder1(inputs)# skip connections for the first network saved
    x = ASPP(x, 64)
    x = decoder1(x, skip_1) 
    outputs1 = output_block(x)

    # Multiplication of the first output and the input
    x = inputs * outputs1

    # U-Net n°2
    x, skip_2 = encoder2(x)
    x = ASPP(x, 64)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = output_block(x)
    
    # General output, we keep 
    outputs = Concatenate()([outputs1, outputs2])

    model = Model(inputs, outputs)
    return model