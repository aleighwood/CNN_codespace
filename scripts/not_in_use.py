import tensorflow as tf
import numpy as np




#fully connected
filter_size = 3
n_C_prev = 3
n_C = 5            


X = np.ones((1,6,6,3))

#weights for standard convolution
W = np.ones((filter_size,filter_size,n_C_prev,n_C))

#weights for DWS convolution
W_depthwise = np.random.rand(filter_size,filter_size,n_C_prev)
W_pointwise = np.random.rand(1,1,n_C_prev,n_C)



#bias weights
b_DWS = np.zeros((1,n_C))

stride = 1
pad = 0

#Y = conv_forward(X,W_dws,b,stride,pad)
#print(f'Z shape: {Y.shape}')

#Y2 = conv_forward_tf(X,W,b,stride,pad)
#print(f'Z2 shape: {Y2.shape}')

#C  =Y2-Y
#print(f'Max difference: {np.max(np.abs(C))}')


Y3_tf = DWS_forward_tf(X, W_depthwise_tf, W_pointwise, b_DWS, stride, pad)
Y3 = DWS_forward(X, W_depthwise, W_pointwise, b_DWS, stride, pad)
print(f'Y3 shape: {Y3.shape}')
print(f'Y3_tf shape: {Y3_tf.shape}')


print(f'Y3 {Y3}')
print(f'Y3_tf {Y3_tf}')


C = Y3 - Y3_tf
print(f'max difference:{np.max(np.abs(C))}')



#what's next? 
# find detailed outline of entire system, inclduding batch norm, re learn batch norm video
# build own python blocks for other modules
# systemVerilog implementation of DWS conv - first need input and output vectors - questa sim
# Relu and batch norm are applied after both depthwise and pointwise layers



# write out blocks
# write out entiresystem

