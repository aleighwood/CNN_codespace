import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    DepthwiseConv2D,
    GlobalAveragePooling2D,
    Reshape,
    Dropout,
    Activation,
)

# Spec used in both TF and numpy versions (pointwise filters, stride for depthwise)
DEPTHWISE_BLOCK_SPECS = [
    (64, (1, 1)),
    (128, (2, 2)),
    (128, (1, 1)),
    (256, (2, 2)),
    (256, (1, 1)),
    (512, (2, 2)),
    (512, (1, 1)),
    (512, (1, 1)),
    (512, (1, 1)),
    (512, (1, 1)),
    (512, (1, 1)),
    (1024, (2, 2)),
    (1024, (1, 1)),
]


def _relu6(x, name):
    return ReLU(max_value=6.0, name=name)(x)


def _conv_block(inputs, filters, kernel_size, strides, block_name):
    """A standard convolution block with Batch Norm and ReLU6."""
    x = Conv2D(filters, kernel_size, padding="same", use_bias=False, strides=strides, name=f"{block_name}_conv")(inputs)
    x = BatchNormalization(name=f"{block_name}_bn")(x)
    x = _relu6(x, name=f"{block_name}_relu6")
    return x


def _depthwise_separable_conv_block(inputs, pointwise_conv_filters, strides, block_idx):
    """A depthwise separable convolution block with ReLU6 activations."""
    # Depthwise convolution
    x = DepthwiseConv2D(
        (3, 3),
        padding="same",
        use_bias=False,
        strides=strides,
        name=f"dws_conv_{block_idx}_depthwise",
    )(inputs)
    x = BatchNormalization(name=f"dws_conv_{block_idx}_dw_bn")(x)
    x = _relu6(x, name=f"dws_conv_{block_idx}_dw_relu6")

    # Pointwise convolution
    x = Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        name=f"dws_conv_{block_idx}_pointwise",
    )(x)
    x = BatchNormalization(name=f"dws_conv_{block_idx}_pw_bn")(x)
    x = _relu6(x, name=f"dws_conv_{block_idx}_pw_relu6")
    return x


def MobileNetFunctional(input_shape=(224, 224, 3), classes=1000, dropout_rate=0.001):
    """
    Builds the MobileNet v1 model using the Keras Functional API.
    This follows the reference architecture (stride pattern and filter counts) and uses ReLU6 activations.
    """
    img_input = Input(shape=input_shape, name="image_input")

    # Initial Convolution Block
    x = _conv_block(img_input, 32, (3, 3), strides=(2, 2), block_name="conv1")

    # Depthwise Separable Blocks
    for idx, (pw_filters, stride) in enumerate(DEPTHWISE_BLOCK_SPECS, start=1):
        x = _depthwise_separable_conv_block(x, pw_filters, strides=stride, block_idx=idx)

    # Classification Head
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Reshape((1, 1, 1024), name="reshape_1")(x)
    x = Dropout(dropout_rate, name="dropout")(x)

    # The final "Dense" layer is implemented as a 1x1 Conv
    x = Conv2D(classes, (1, 1), padding="same", name="conv_preds")(x)
    x = Reshape((classes,), name="reshape_2")(x)
    predictions = Activation("softmax", name="act_softmax")(x)

    # Create model
    model = tf.keras.Model(inputs=img_input, outputs=predictions, name="MobileNetV1")
    return model


def export_weights_to_numpy_dict(model):
    """
    Export weights from the Keras MobileNetV1 model into the parameter dict
    expected by MobileNet_np.MobileNetV1Numpy.
    """
    params = {}

    # Conv1
    conv_layer = model.get_layer("conv1_conv")
    bn_layer = model.get_layer("conv1_bn")
    W_conv = conv_layer.get_weights()[0]  # (3,3,3,32)
    params["conv1"] = {
        "W": W_conv,
        "b": np.zeros(W_conv.shape[-1], dtype=W_conv.dtype),
        "gamma": bn_layer.gamma.numpy(),
        "beta": bn_layer.beta.numpy(),
        "moving_mean": bn_layer.moving_mean.numpy(),
        "moving_var": bn_layer.moving_variance.numpy(),
    }

    # Depthwise-separable blocks
    for idx, (pw_filters, _) in enumerate(DEPTHWISE_BLOCK_SPECS, start=1):
        dw_layer = model.get_layer(f"dws_conv_{idx}_depthwise")
        bn_dw = model.get_layer(f"dws_conv_{idx}_dw_bn")
        pw_layer = model.get_layer(f"dws_conv_{idx}_pointwise")
        bn_pw = model.get_layer(f"dws_conv_{idx}_pw_bn")

        # Depthwise weights: (3,3,in_c,1) -> (3,3,in_c)
        W_depthwise = dw_layer.get_weights()[0][..., 0]

        params[f"ds_{idx}"] = {
            "W_depthwise": W_depthwise,
            "gamma_dw": bn_dw.gamma.numpy(),
            "beta_dw": bn_dw.beta.numpy(),
            "moving_mean_dw": bn_dw.moving_mean.numpy(),
            "moving_var_dw": bn_dw.moving_variance.numpy(),
            "W_pointwise": pw_layer.get_weights()[0],
            "b_pointwise": np.zeros(pw_filters, dtype=W_depthwise.dtype),
            "gamma_pw": bn_pw.gamma.numpy(),
            "beta_pw": bn_pw.beta.numpy(),
            "moving_mean_pw": bn_pw.moving_mean.numpy(),
            "moving_var_pw": bn_pw.moving_variance.numpy(),
        }

    # FC / final conv preds
    conv_preds = model.get_layer("conv_preds")
    W_fc, b_fc = conv_preds.get_weights()
    params["fc"] = {
        # reshape (1,1,1024,classes) -> (1024, classes)
        "W": W_fc.reshape(W_fc.shape[2], W_fc.shape[3]),
        "b": b_fc.reshape(-1, 1),
    }

    return params


if __name__ == "__main__":
    mobilenet_functional = MobileNetFunctional()
    mobilenet_functional.summary()
