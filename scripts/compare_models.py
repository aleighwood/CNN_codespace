import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MobileNet_tf import MobileNetFunctional, export_weights_to_numpy_dict
from src.MobileNet_np import mobilenet_v1_numpy_forward

np.random.seed(0)
tf.random.set_seed(0)

# Build TF model and export weights
model = MobileNetFunctional(input_shape=(224,224,3), classes=5)
params = export_weights_to_numpy_dict(model)

# Shared random input
x = np.random.randn(1, 224, 224, 3).astype(np.float32)

# TF forward
y_tf = model(x, training=False).numpy()

# Numpy forward
y_np = mobilenet_v1_numpy_forward(x, params)

print("TF shape:", y_tf.shape, "NumPy shape:", y_np.shape)
print("Max abs diff:", np.max(np.abs(y_tf - y_np)))
print("Row sums TF:", y_tf.sum(axis=1), "Row sums NumPy:", y_np.sum(axis=1))
