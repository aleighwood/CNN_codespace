import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Reshape, Dropout, Activation

def _conv_block(inputs, filters, kernel_size, strides):
    """A standard convolution block with Batch Norm and ReLU."""
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False, strides=strides, name=f'conv_{filters}')(inputs)
    x = BatchNormalization(name=f'conv_{filters}_bn')(x)
    x = ReLU(name=f'conv_{filters}_relu')(x)
    return x

def _depthwise_separable_conv_block(inputs, pointwise_conv_filters, strides):
    """A depthwise separable convolution block."""
    # Depthwise convolution
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False, strides=strides, name=f'dws_conv_{pointwise_conv_filters}_depthwise')(inputs)
    x = BatchNormalization(name=f'dws_conv_{pointwise_conv_filters}_dw_bn')(x)
    x = ReLU(name=f'dws_conv_{pointwise_conv_filters}_dw_relu')(x)

    # Pointwise convolution
    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name=f'dws_conv_{pointwise_conv_filters}_pointwise')(x)
    x = BatchNormalization(name=f'dws_conv_{pointwise_conv_filters}_pw_bn')(x)
    x = ReLU(name=f'dws_conv_{pointwise_conv_filters}_pw_relu')(x)
    return x

def MobileNetFunctional(input_shape=(224, 224, 3), classes=1000):
    """
    Builds the MobileNet v1 model using the Keras Functional API.
    """
    img_input = Input(shape=input_shape)

    # Initial Convolution Block
    x = _conv_block(img_input, 32, (3, 3), strides=(2, 2))

    # Depthwise Separable Blocks
    x = _depthwise_separable_conv_block(x, 64, strides=(1, 1))
    x = _depthwise_separable_conv_block(x, 128, strides=(2, 2))
    x = _depthwise_separable_conv_block(x, 128, strides=(1, 1))
    x = _depthwise_separable_conv_block(x, 256, strides=(2, 2))
    x = _depthwise_separable_conv_block(x, 256, strides=(1, 1))
    x = _depthwise_separable_conv_block(x, 512, strides=(2, 2))
    
    # Five stacked blocks
    x = _depthwise_separable_conv_block(x, 512, strides=(1, 1))
    x = _depthwise_separable_conv_block(x, 512, strides=(1, 1))
    x = _depthwise_separable_conv_block(x, 512, strides=(1, 1))
    x = _depthwise_separable_conv_block(x, 512, strides=(1, 1))
    x = _depthwise_separable_conv_block(x, 512, strides=(1, 1))

    x = _depthwise_separable_conv_block(x, 1024, strides=(2, 2))
    x = _depthwise_separable_conv_block(x, 1024, strides=(1, 1))

    # Classification Head
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(0.001, name='dropout')(x)
    
    # The final "Dense" layer is implemented as a 1x1 Conv
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Reshape((classes,), name='reshape_2')(x)
    predictions = Activation('softmax', name='act_softmax')(x)

    # Create model
    model = tf.keras.Model(inputs=img_input, outputs=predictions, name='MobileNetV1')
    
    return model

# Create and summarize the model
mobilenet_functional = MobileNetFunctional()
mobilenet_functional.summary()