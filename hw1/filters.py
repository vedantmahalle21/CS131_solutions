"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for m in range(Hi):
    	for n in range(Wi):
    		for i in range(Hk):
    			for j in range(Wk):
    				if(m+1-i>= Hi or n+1-j>= Wi or m+1-i < 0 or n+1-j < 0 ):
    					out[m][n] += 0
    				else:
    					out[m][n] += kernel[i][j]*image[m+1-i][n+1-j]
    		 

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    h_ = np.zeros((H + 2*pad_height, W + 2*pad_width))
    h_[pad_height:H +pad_height,pad_width:W + pad_width] = image
    out = h_
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    image = zero_pad(image, Hk//2, Wk//2)
    kernel = np.flip(kernel,0)
    kernel = np.flip(kernel,1)
    for m in range(Hi):
    	for n in range(Wi):
    		out[m,n] = np.sum(image[m:m+Hk,n:n+Wk]*kernel)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(g,0)
    g = np.flip(g,1)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_mean = np.sum(g)/np.size(g)
    g = g - g_mean
    g = np.flip(g,0)
    g = np.flip(g,1)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    if g.shape[0] % 2 == 0:
        g = g[0:-1]
    if g.shape[1] % 2 == 0:
        g = g[:,0:-1]
    Hf, Wf = np.shape(f)
    Hg, Wg = np.shape(g)
    ### YOUR CODE HERE

    g_mean = np.sum(g)/np.size(g)
    g_std = np.std(g)
    g = (g - g_mean)/g_std
    out = np.zeros((Hf, Wf))
    for m in range(Hg//2,Hf-Hg//2):
    	for n in range(Wg//2, Wf-Wg//2):
    		f_mean = np.sum(f[m-Hg//2:m+Hg//2+1,n-Wg//2:n+Wg//2+1])/np.size(f[m-Hg//2:m+Hg//2+1,n-Wg//2:n+Wg//2+1])
    		f_std = np.std(f[m-Hg//2:m+Hg//2+1,n-Wg//2:n+Wg//2+1])
    		f[m-Hg//2:m+Hg//2+1,n-Wg//2:n+Wg//2+1] = (f[m-Hg//2:m+Hg//2+1,n-Wg//2:n+Wg//2+1]- f_mean)/f_std
    		out[m,n] = np.sum(f[m-Hg//2:m+Hg//2+1,n-Wg//2:n+Wg//2+1]*g)
    ### END YOUR CODE

    return out