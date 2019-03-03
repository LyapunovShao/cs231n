import numpy as np
from random import shuffle


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
    num_train = X.shape[0]
    num_classes = W.shape[1]
    num_dim = W.shape[0]
    S = X.dot(W)
    S -= np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S)
    exp_S_sum = np.sum(exp_S, axis=1)
    for i in range(num_train):
        loss -= np.log(exp_S[i, y[i]] / exp_S_sum[i])
        dW[:, y[i]] -= X[i, :]
        dW += X[i, :].reshape((num_dim, 1)).dot(exp_S[i, :].reshape((1, num_classes))) / exp_S_sum[i]
    loss /= num_train
    loss += np.sum(reg * W * W)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

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
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    num_dim = W.shape[0]
    S = X.dot(W)
    exp_S = np.exp(X.dot(W))
    exp_S_sum = np.sum(exp_S, axis=1, keepdims=True)
    position = np.zeros((num_train, num_classes))
    position[np.arange(num_train), y] = 1
    loss += -np.sum(S[np.arange(num_train), y]) + np.sum(np.log(exp_S_sum))
    loss /= num_train
    loss += np.sum(reg * W * W)
    S -= np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(X.dot(W))
    exp_S_sum = np.sum(exp_S, axis=1, keepdims=True)
    dW += -X.T.dot(position) + X.T.dot(exp_S / (exp_S_sum - np.zeros_like(exp_S)))
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

