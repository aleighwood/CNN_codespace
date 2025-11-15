import tensorflow as tf
import numpy as np


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),mode = 'constant', constant_values = (0,0)) 
    #X_pad = np.pad(X,((pad,pad),(pad,pad)),mode = 'constant', constant_values = (0,0)) 

 
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    s= a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)

    return Z

def conv_forward(A_prev, W, b, stride,pad):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev) 
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = np.shape(W)
    
    # Retrieve information from "hparameters" (≈2 lines)
    #stride = hparameters["stride"]
    #pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    n_H = int((n_H_prev - f + 2*pad)/stride)+1
    n_W = int((n_W_prev - f + 2*pad)/stride)+1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m,n_H,n_W,n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev,pad)
    
    #loop through all data
    for i in range(m):
        #select ith image 
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            #start and end of current vertical slice
            vert_start = h*stride
            vert_end = h*stride+f
            
            for w in range(n_W):
                #start and end of current horizontal slice
                horiz_start = w*stride
                horiz_end = w*stride+f
                
                #loop through all filters nc', 3D block convolution each time
                for c in range(n_C):
                    #select slice
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    weights = W[:,:,:,c]

                    biases = b[:,:,:,c]

                    Z[i,h,w,c] = conv_single_step(a_slice_prev,weights,biases)
                    
    return Z

def conv_forward_tf(A,W,b,stride,pad):
    """
    Implements the forward propagation for a single convolution layer using TensorFlow's Functional API.
    
    Arguments:
    A -- input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    stride -- integer, stride of the convolution
    pad -- integer, amount of zero-padding
        
    Returns:
    Z -- conv output, TensorFlow tensor of shape (m, n_H, n_W, n_C)
    """
    # Get input shape and parameters
    input_shape = A.shape[1:]
    n_C = W.shape[3]
    f = W.shape[0]

    # Define the input tensor
    X_input = tf.keras.layers.Input(input_shape)

    # pad input 
    X_padded = tf.keras.layers.ZeroPadding2D((pad, pad))(X_input)
    Z = tf.keras.layers.Conv2D(filters=n_C, kernel_size=(f, f), strides=(stride, stride), padding='valid', name='conv1')(X_padded)

    #create model
    model = tf.keras.Model(inputs=X_input, outputs=Z)

    #set own weights
    model.get_layer('conv1').set_weights([W, b.reshape(-1)])

    #run forward pass 
    output = model(A)

    return output


#next: build a Depthwise seperable convolution function
def DWS_forward(A,W_dws,b,stride,pad):
    """
    Implements the forward propagation for a DWS convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev) 
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = np.shape(W)
    
    # Retrieve information from "hparameters" (≈2 lines)
    #stride = hparameters["stride"]
    #pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    n_H = int((n_H_prev - f + 2*pad)/stride)+1
    n_W = int((n_W_prev - f + 2*pad)/stride)+1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m,n_H,n_W,n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev,pad)
    
    #loop through all data
    for i in range(m):
        #select ith image 
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            #start and end of current vertical slice
            vert_start = h*stride
            vert_end = h*stride+f
            
            for w in range(n_W):
                #start and end of current horizontal slice
                horiz_start = w*stride
                horiz_end = w*stride+f
                
                #loop through all filters nc', 3D block convolution each time
                for c in range(n_C):
                    #select slice
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    weights = W[:,:,:,c]

                    biases = b[:,:,:,c]

                    Z[i,h,w,c] = conv_single_step(a_slice_prev,weights,biases)
                    
    return Z





X = np.ones((1,224,224,3))
W = np.ones((3,3,3,32))
b = np.zeros((1,1,1,32))
stride = 2
pad = 1
Y = conv_forward(X,W,b,stride,pad)
print(f'Z shape: {Y.shape}')

Y2 = conv_forward_tf(X,W,b,stride,pad)
print(f'Z2 shape: {Y2.shape}')

C  =Y2-Y
print(f'Max difference: {np.max(np.abs(C))}')








