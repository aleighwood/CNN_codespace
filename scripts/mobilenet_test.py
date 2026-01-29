# test_mobilenet_np.py
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MobileNet_np import DEPTHWISE_BLOCK_SPECS, mobilenet_v1_numpy_forward

def make_params(num_classes=5):
    params={}
    params['conv1']=dict(
        W=np.random.randn(3,3,3,32).astype(np.float32),
        b=np.zeros(32,dtype=np.float32),
        gamma=np.ones(32,dtype=np.float32),
        beta=np.zeros(32,dtype=np.float32),
        moving_mean=np.zeros(32,dtype=np.float32),
        moving_var=np.ones(32,dtype=np.float32),
    )
    c_in=32
    for name,pw_filters,stride in DEPTHWISE_BLOCK_SPECS:
        params[name]=dict(
            W_depthwise=np.random.randn(3,3,c_in).astype(np.float32),
            gamma_dw=np.ones(c_in,dtype=np.float32),
            beta_dw=np.zeros(c_in,dtype=np.float32),
            moving_mean_dw=np.zeros(c_in,dtype=np.float32),
            moving_var_dw=np.ones(c_in,dtype=np.float32),
            W_pointwise=np.random.randn(1,1,c_in,pw_filters).astype(np.float32),
            b_pointwise=np.zeros(pw_filters,dtype=np.float32),
            gamma_pw=np.ones(pw_filters,dtype=np.float32),
            beta_pw=np.zeros(pw_filters,dtype=np.float32),
            moving_mean_pw=np.zeros(pw_filters,dtype=np.float32),
            moving_var_pw=np.ones(pw_filters,dtype=np.float32),
        )
        c_in = pw_filters
    params['fc']=dict(
        W=np.random.randn(1024,num_classes).astype(np.float32),
        b=np.zeros((num_classes,1),dtype=np.float32),
    )
    return params

if __name__ == "__main__":
    x = np.random.randn(1, 224, 224, 3).astype(np.float32)  # paper input size
    params = make_params(num_classes=5)
    y = mobilenet_v1_numpy_forward(x, params)
    print("Output shape:", y.shape)
    print("Row sums (should be ~1):", y.sum(axis=1))
