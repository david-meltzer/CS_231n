from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    ind_zero = np.zeros((X.shape[0],W.shape[1]))
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                ind_zero[i,j]=0
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                ind_zero[i,j]=1
            else:
                ind_zero[i,j]=0

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for p in range(W.shape[0]):
      for q in range (W.shape[1]):
        for i in range(num_train):  
            dW[p,q] += 1/num_train*ind_zero[i,q]*X[i,p]
            if q == y[i]:
              for j in range(num_classes):
                dW[p,q] -= 1/num_train*ind_zero[i,j]*X[i,p]
        dW[p,q] += 2*reg*W[p,q] 

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
      # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    
    #correct_class_score = scores[:,y]
    correct_class_score = np.asarray([scores[i,y[i]] for i in range(num_train)])
    
    correct_class_score = correct_class_score.reshape(correct_class_score.shape[0],1)
    
    loss = np.sum(1/num_train*(np.maximum(0,scores-correct_class_score+1)))-1+reg*np.sum(np.square(W))
    
    ind_zero=np.heaviside((np.maximum(0,scores-correct_class_score+1)),0)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #indyq
    grad1 = 1/num_train*X.transpose().dot(ind_zero)
    gradreg = 2*reg*W

    indvec=np.sum(ind_zero,axis=1,keepdims=True)
    
    Xrescaled = -1/num_train*indvec*X
  
    ind_matrix = np.zeros((num_train,num_classes))

    for i in range(num_train):
      ind_matrix[i,y[i]]=1
  

    grad2=np.dot(Xrescaled.transpose(),ind_matrix)

    dW=grad1 + grad2 + gradreg
    

    #dSdW=np.tensordot(X,np.identity(num_classes))
    #dScorrectdW= dSdW[:,:,y,:]



    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
