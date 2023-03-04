from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_test = X.shape[0]
    num_classes = W.shape[1] 
    #losslog = 0.0
    f = np.dot(X,W)
    fmax = np.amax(f,axis=1,keepdims=True)
    f = f - fmax

    for i in range(num_test):
      losslog=0.0
      loss += -f[i,y[i]]/num_test
      for c in range(num_classes):
        losslog += np.exp(f[i,c])
      loss += np.log(losslog)/num_test

    for i1 in range(W.shape[0]):
      for i2 in range(W.shape[1]):
        loss += reg*np.square(W[i1,i2])

    denom_sum=np.zeros((num_test,1))

    for i4 in range(num_test):
      for c4 in range(num_classes):
        denom_sum[i4] += np.exp(f[i4,c4])

    #print(denom_sum[:3])

    for p in range(W.shape[0]):
      for q in range(W.shape[1]):
        for i3 in range(num_test):
          #for c2 in range(num_classes):
          #  denom_sum +=np.exp(f[i3,c2])
          #dW[p,q] += 1/denom_sum*np.exp(f[i3,q])*X[i3,p]
          dW[p,q] += 1/denom_sum[i3]*np.exp(f[i3,q])*X[i3,p]/num_test
          if q == y[i3]:
            dW[p,q] += -X[i3,p]/num_test
        dW[p,q] += 2*reg*W[p,q] 

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_test = X.shape[0]
    num_classes = W.shape[1]
    f = np.dot(X,W)
    fmax = np.amax(f,axis=1,keepdims=True)
    f = f - fmax

    fdiag = np.asarray([f[i,y[i]] for i in range(num_test)])

    Lpt1 = -np.sum(fdiag)/num_test

    Lpt2 = 1/num_test*np.sum(np.log(np.sum(np.exp(f),axis=1,keepdims=True)))

    Lpt3 = reg*np.sum(np.square(W))

    loss = Lpt1 + Lpt2 + Lpt3

    dyq = np.zeros((num_test,num_classes))

    for i in range(num_test):
      dyq[i,y[i]]=1.0

    gradpt1 = -1*1/num_test*np.dot(X.transpose(),dyq)

    rescaled_f= 1/(np.sum(np.exp(f),axis=1,keepdims=True))*np.exp(f)

    gradpt2 = 1/num_test*np.dot(X.transpose(),rescaled_f)

    gradpt3 = 2*reg*W
    

    dW =gradpt1+gradpt2+gradpt3

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
