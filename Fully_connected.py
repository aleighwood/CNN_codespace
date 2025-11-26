import numpy as np
import tensorflow as tf

def FC_forward_prop(A_prev,weights,bias):
    """
    Forward propagation of a fully conneced NN 
    
    Argument:
    A_prev -- output of previous layer (m,N_prev,1)
    bias -- bias weights (N,1)
    weights -- weights to be used in the NN (N_prev,N)

    
    Returns:
    A -- output of NN (m,N,1)
    """
    # FC layer in MobileNet has no activation, input = 1024, output = 1000

    #find 
    N_prev,N = np.shape(weights)
    m = np.shape(A_prev)[0]

            # initialise A 
    A = np.zeros((m,N,1))

    for i in range(m):


        A[i] = bias + np.dot(weights.T,A_prev[i,:])
        
    return A

def FC_forward_prop_tf(A_prev,weights,bias):
    """
    Forward propagation of a fully conneced NN using TF
    
    Argument:
    A_prev -- output of previous layer (m,N_prev)
    bias -- bias weights (N,1)
    weights -- weights to be used in the NN (N_prev,N)

    
    Returns:
    A -- output of NN (m,N,1)
    """
        
    input_features = np.shape(A_prev)[1]
    output_units = np.shape(weights)[1]

    #input layer
    X_input = tf.keras.layers.Input(shape = (input_features,))

    #dense layer
    Z = tf.keras.layers.Dense(units = output_units, name = 'dense1',use_bias = True)(X_input)

    #create model 
    model = tf.keras.Model(inputs = X_input, outputs = Z)

    #set own weights
    model.get_layer('dense1').set_weights([weights,bias.reshape(-1)])

    #run forward pass
    output = model(A_prev)

    return output




m =1
N_prev = 4
N = 3
bias = np.ones((N,1))
A_prev = np.ones((m,N_prev,1))
weights = np.ones((N_prev,N))

Z_dense = FC_forward_prop(A_prev,weights,bias)
Z_dense_tf = FC_forward_prop_tf(A_prev,weights,bias)

error = np.max(np.abs(Z_dense-Z_dense_tf))
print(error)

print(f'tf output shape: {np.shape(Z_dense)}')
print(f'tf output shape: {np.shape(Z_dense_tf)}')




