#Depthwise seperable convolution function
def depthwise_conv_tf(A, W_depthwise, W_pointwise, b, stride, pad):
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

def depthwise_conv(A,W_depthwise,W_pointwise,b,stride,pad):
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
