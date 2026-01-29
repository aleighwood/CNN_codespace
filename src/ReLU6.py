import numpy as np

def relu6(x):
    return np.minimum(np.maximum(x, 0), 6)
