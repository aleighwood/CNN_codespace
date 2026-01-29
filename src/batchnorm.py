import numpy as np

def batchnorm_inference(x, gamma, beta, moving_mean, moving_var, eps=1e-3):
    """
    x:           Input tensor of shape (m, H, W, C)
    gamma:       Scale parameter of shape (C,)
    beta:        Shift parameter of shape (C,)
    moving_mean: Running mean of shape (C,)
    moving_var:  Running variance of shape (C,)
    eps:         Small constant for numerical stability
    """

    # Reshape parameters for broadcasting
    gamma = gamma.reshape(1, 1, 1, -1)
    beta  = beta.reshape(1, 1, 1, -1)
    mean  = moving_mean.reshape(1, 1, 1, -1)
    var   = moving_var.reshape(1, 1, 1, -1)

    # Normalize
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Scale + shift
    out = gamma * x_norm + beta

    return out