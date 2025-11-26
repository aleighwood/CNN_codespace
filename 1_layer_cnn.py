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

def conv_single_step(a_slice_prev, W, b, add_bias=True):
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

    if add_bias == True:
        Z = Z + float(b)
        

    return Z

#standard forward conv
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


#Depthwise seperable convolution function
def DWS_forward_tf(A, W_depthwise, W_pointwise, b, stride, pad):
    """
    Implements the forward propagation for a depthwise separable convolution layer using TensorFlow.
    
    Arguments:
    A -- input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W_depthwise -- Depthwise weights, numpy array of shape (f, f, n_C_prev, 1)
    W_pointwise -- Pointwise weights, numpy array of shape (1, 1, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    stride -- integer, stride of the convolution
    pad -- integer, amount of zero-padding
        
    Returns:
    Z -- conv output, TensorFlow tensor
    """
    input_shape = A.shape[1:]
    n_C = W_pointwise.shape[3]
    f = W_depthwise.shape[0]

    X_input = tf.keras.layers.Input(input_shape)
    
    # Padding is applied separately to match manual calculations
    X = tf.keras.layers.ZeroPadding2D((pad, pad))(X_input)

    # 1. Depthwise Convolution (no bias here)
    depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(f, f), 
        strides=(stride, stride), 
        padding='valid', 
        use_bias=False, # Bias is typically applied after the pointwise step
        name='depthwise_conv'
    )
    
    # 2. Pointwise Convolution (1x1 convolution with bias)
    pointwise_conv = tf.keras.layers.Conv2D(
        filters=n_C, 
        kernel_size=(1, 1), 
        strides=(1, 1), 
        padding='valid',
        use_bias=True,
        name='pointwise_conv'
    )

    # Chain the layers
    depthwise_output = depthwise_conv(X)
    pointwise_output = pointwise_conv(depthwise_output)

    model = tf.keras.Model(inputs=X_input, outputs=pointwise_output)

    # Set the weights for both layers
    model.get_layer('depthwise_conv').set_weights([W_depthwise])
    model.get_layer('pointwise_conv').set_weights([W_pointwise, b.reshape(-1)])

    output = model(A)
    
    return output

def DWS_forward(A,W_depthwise,W_pointwise,b,stride,pad):
    """
    Implements the forward propagation for a DWS convolution function
    
    Arguments:
    A -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W_depthwise -- Weights for depthwise convolution, numpy array of shape (f, f, n_C_prev)
    W_pointwise -- Weights for pointwise convolution, numpy array of shape (1,1,n_C_prev,n_C)
    b -- Biases, numpy array of shape (1,n_C)
    stride
    pad
        
    Returns:
    output -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    
    """
    
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A) 
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev) = np.shape(W_depthwise)
    n_C = np.shape(W_pointwise)[3]
    f_pointwise = 1
    

    print(f'N_C_prev: {n_C_prev}')
    print(f'A size: {np.shape(A)}')
    # Retrieve information from "hparameters" (≈2 lines)
    #stride = hparameters["stride"]
    #pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    n_H = int((n_H_prev - f + 2*pad)/stride)+1
    n_W = int((n_W_prev - f + 2*pad)/stride)+1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m,n_H,n_W,n_C_prev))
    output = np.zeros((m,n_H,n_W,n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A,pad)
    
    #perform depthwise convolution
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
                
                #loop through all filters nc, 2D convolution each time
                for j in range(n_C_prev):
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,j]

                    weights = W_depthwise[:,:, j]
                    #no bias after depthwise
                    biases = 0
                    Z[i,h,w,j] = conv_single_step(a_slice_prev,weights,biases, False)

    # perform pointwise

    for i in range(m):
        #select image 
        Z_prev = Z[i,:,:,:]
        for h in range(n_H):
            vert_start = h
            vert_end = h+f_pointwise
            for w in range(n_W):
                #start and end of current horizontal slice
                horiz_start = w
                horiz_end = w+f_pointwise
                for p in range(n_C):
                    #take whole 'depth' column
                    DWS_prev = Z_prev[vert_start:vert_end,horiz_start:horiz_end,:]
                    weights = W_pointwise[:,:,:,p]
                    biases = b[:,p]

                    output[i,h,w,p] = conv_single_step(DWS_prev,weights,biases,True)

    return output
        
#Average pooling 

    """
    Implements average pooling layer
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filter -- filter size, int
    stride -- stride, int
    pad -- pad, int

        
    Returns:
    Z -- avg pool, numpy array of shape (m, n_H, n_W, n_C)
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

#pooling layer
def pool_forward(A_prev, f,stride, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- filter size 
    stride -- stride of filter
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    
    
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    

    for i in range(m):
    
        for h in range(n_H):
            vert_start = h*stride
            vert_end = h*stride+f

            for w in range(n_W):
                horiz_start = w*stride
                horiz_end = w*stride+f

                for c in range(n_C):
                    a_slice_prev = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]

                    if mode == "max":
                        A[i,h,w,c] = np.max(a_slice_prev)

                    elif mode == "average":
                        A[i,h,w,c] = np.mean(a_slice_prev)
    
    print(f"A shape: {np.shape(A)}")

    
    return A

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

#weights for keras implementation
W_depthwise_tf = W_depthwise[...,np.newaxis]

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

