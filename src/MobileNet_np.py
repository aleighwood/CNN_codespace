import numpy as np

from .batchnorm import batchnorm_inference
from .conv import conv
from .depthwise_conv import depthwise_conv_only
from .Fully_connected import fully_connected
from .ReLU6 import relu6
from .pooling import pooling
from .softmax import softmax_numpy

# (name, pointwise_filters, stride_for_depthwise)
DEPTHWISE_BLOCK_SPECS = [
    ("ds_1", 64, 1),
    ("ds_2", 128, 2),
    ("ds_3", 128, 1),
    ("ds_4", 256, 2),
    ("ds_5", 256, 1),
    ("ds_6", 512, 2),
    ("ds_7", 512, 1),
    ("ds_8", 512, 1),
    ("ds_9", 512, 1),
    ("ds_10", 512, 1),
    ("ds_11", 512, 1),
    ("ds_12", 1024, 2),
    ("ds_13", 1024, 1),
]


class MobileNetV1Numpy:
    """
    Numpy implementation of MobileNet v1 inference following the layout in the paper:
    - Initial 3x3 conv (stride 2) + BN + ReLU6
    - 13 depthwise-separable blocks (each: depthwise 3x3 + BN + ReLU6, then pointwise 1x1 + BN + ReLU6)
    - Global average pool
    - Fully-connected (1x1 conv equivalent) to class logits
    - Softmax

    Parameters are supplied via a dict. Expected keys:
      conv1: {
          "W": (3,3,3,32), "b": optional bias,
          "gamma","beta","moving_mean","moving_var"
      }
      ds_k for k in 1..13: {
          "W_depthwise": (3,3,C_in),
          "gamma_dw","beta_dw","moving_mean_dw","moving_var_dw",
          "W_pointwise": (1,1,C_in,C_out), "b_pointwise": optional,
          "gamma_pw","beta_pw","moving_mean_pw","moving_var_pw"
      }
      fc: {"W": (1024, num_classes), "b": (num_classes,1)}
    """

    def __init__(self, params, eps=1e-3):
        self.params = params
        self.eps = eps

    @staticmethod
    def _format_conv_bias(bias, channels):
        """Ensure conv bias is shaped (1,1,1,C)."""
        if bias is None:
            return np.zeros((1, 1, 1, channels))
        bias_arr = np.array(bias)
        if bias_arr.size == channels:
            bias_arr = bias_arr.reshape(1, 1, 1, channels)
        return bias_arr

    def _conv_bn_relu(self, x, name, stride, pad):
        layer = self.params[name]
        W = layer["W"]
        b = self._format_conv_bias(layer.get("b"), W.shape[-1])
        z = conv(x, W, b, stride=stride, pad=pad)
        z = batchnorm_inference(
            z,
            gamma=layer["gamma"],
            beta=layer["beta"],
            moving_mean=layer["moving_mean"],
            moving_var=layer["moving_var"],
            eps=self.eps,
        )
        return relu6(z)

    def _depthwise_separable_block(self, x, name, stride, pad, expected_pw_filters):
        layer = self.params[name]

        dw = depthwise_conv_only(x, layer["W_depthwise"], stride=stride, pad=pad)
        dw = batchnorm_inference(
            dw,
            gamma=layer["gamma_dw"],
            beta=layer["beta_dw"],
            moving_mean=layer["moving_mean_dw"],
            moving_var=layer["moving_var_dw"],
            eps=self.eps,
        )
        dw = relu6(dw)

        pw_filters = layer["W_pointwise"].shape[-1]
        if expected_pw_filters is not None and pw_filters != expected_pw_filters:
            raise ValueError(f"{name} pointwise filters mismatch: expected {expected_pw_filters}, got {pw_filters}")

        b_pw = self._format_conv_bias(layer.get("b_pointwise"), pw_filters)
        pw = conv(dw, layer["W_pointwise"], b_pw, stride=1, pad=0)
        pw = batchnorm_inference(
            pw,
            gamma=layer["gamma_pw"],
            beta=layer["beta_pw"],
            moving_mean=layer["moving_mean_pw"],
            moving_var=layer["moving_var_pw"],
            eps=self.eps,
        )
        pw = relu6(pw)
        return pw

    @staticmethod
    def _global_average_pool(x):
        _, n_H, _, _ = x.shape
        return pooling(x, f=n_H, stride=1, mode="average")

    def forward(self, x, return_logits=False):
        """
        Run inference. `x` should be float32/float64 numpy array (m, H, W, 3).
        """
        out = self._conv_bn_relu(x, "conv1", stride=2, pad=1)

        for name, pw_filters, stride in DEPTHWISE_BLOCK_SPECS:
            out = self._depthwise_separable_block(out, name, stride=stride, pad=1, expected_pw_filters=pw_filters)

        out = self._global_average_pool(out)
        m = out.shape[0]
        out = out.reshape(m, -1, 1)  # (m, 1024, 1)

        fc_params = self.params["fc"]
        logits = fully_connected(out, fc_params["W"], fc_params["b"])
        logits = logits.reshape(m, -1)

        if return_logits:
            return logits

        return softmax_numpy(logits, axis=1)


def mobilenet_v1_numpy_forward(x, params, return_logits=False, eps=1e-3):
    """
    Convenience function to run MobileNet v1 with a parameter dict.
    """
    model = MobileNetV1Numpy(params=params, eps=eps)
    return model.forward(x, return_logits=return_logits)
