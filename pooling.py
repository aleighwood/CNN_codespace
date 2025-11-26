
def pooling(A_prev, f,stride, mode = "max"):
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
