from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    pre_next_h = (np.dot(prev_h, Wh)+ b) + np.dot(x, Wx)
    next_h = np.tanh(pre_next_h)
    cache = (pre_next_h, x, prev_h, Wx, Wh, b)
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    pre_next_h, x, prev_h, Wx, Wh, b = cache
    dtanh = 1-np.tanh(pre_next_h)**2
    dout = dnext_h * dtanh
    dx =  np.dot(dout, Wx.T)
    dprev_h = np.dot(dout, Wh.T)
    dWx = np.dot(x.T, dout)
    dWh = np.dot(prev_h.T, dout)
    db = np.sum(dout, axis=0)
    #######################*#######################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """

    N, T, D = x.shape
    H = h0.shape[1]
    h = np.zeros((N, T, H))
    next_h = h0
    cache = list()
    for i in range(T):
        cur_x = x[:,i,:]
        next_h, cur_cache = rnn_step_forward(cur_x, next_h, Wx, Wh, b)
        h[:,i,:] = next_h
        cache.append(cur_cache)

    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    pre_next_h, x, prev_h, Wx, Wh, b = cache[0]
    N, T, H = dh.shape
    D=x.shape[1]
    dx = np.zeros((N,T,D))
    dh0 = np.zeros((N,H))
    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    db = np.zeros(H)

    cur_cache = cache[T-1]
    dnext_h = dh[:,T-1,:]
    cur_dx, cur_dprev_h1, cur_dWx, cur_dWh, cur_db = rnn_step_backward(dnext_h, cur_cache)
    dx[:, T-1, :] = cur_dx
    dWx += cur_dWx
    dWh += cur_dWh
    db += cur_db
    dh0 += cur_dprev_h1
    for i in range(T - 2, -1, -1):
        cur_cache = cache[i]
        cur_dx, cur_dprev_h1, cur_dWx, cur_dWh, cur_db = rnn_step_backward(cur_dprev_h1, cur_cache)
        dx[:, i, :] += cur_dx
        dWx += cur_dWx
        dWh += cur_dWh
        db += cur_db
        #dh0 += cur_dprev_h1

        dnext_h = dh[:, i, :]
        cur_dx, cur_dprev_h2, cur_dWx, cur_dWh, cur_db = rnn_step_backward(dnext_h, cur_cache)
        #dh0 += cur_dprev_h2
        dx[:, i, :] += cur_dx
        dWx += cur_dWx
        dWh += cur_dWh
        db += cur_db

        cur_dprev_h1 +=cur_dprev_h2
    dh0 = cur_dprev_h1

    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N,T = x.shape
    V,D = W.shape
    out = np.zeros((N, T, D))
    for i in range(N):
        for j in range(T):
            out[i,j,:] = W[x[i,j]]
    cache = (x, W)
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    N,T,D = dout.shape
    x, W = cache
    V, D = W.shape
    dW = np.zeros((V, D))
    for i in range(N):
        for j in range(T):
            dW[x[i, j]] += dout[i,j,:]


    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    N, D = x.shape
    H = prev_h.shape[1]
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    ai = a[:,0:H]
    af = a[:,H:2*H]
    ao = a[:, 2*H:3*H]
    ag = a[:,3*H:4*H]
    i= sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)

    next_c = f * prev_c + i*g
    next_h = o * np.tanh(next_c)

    cache = (x, prev_h, prev_c, next_c, next_h, Wx, Wh, b, i, f, o, g, ai, af, ao, ag)

    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    x, prev_h, prev_c, next_c, next_h, Wx, Wh, b, i, f, o, g, ai, af, ao, ag = cache
    N, H = dnext_h.shape
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    dprev_c = f * (dnext_c + o * dnext_h * (1-np.tanh(next_c)**2))
    do = dnext_h * np.tanh(next_c)
    di = g * (dnext_c + o * dnext_h * (1-np.tanh(next_c)**2))
    dg = i * (dnext_c + o * dnext_h * (1-np.tanh(next_c)**2))
    df = prev_c * (dnext_c + o * dnext_h * (1-np.tanh(next_c)**2))

    ones = np.ones((N, H))
    dai = di * sigmoid(ai) * (ones - sigmoid(ai))
    daf = df * sigmoid(af) * (ones - sigmoid(af))
    dag = dg * (1-np.tanh(ag)**2)
    dao = do * sigmoid(ao) * (ones - sigmoid(ao))

    da = np.append(dai, daf, axis=1)
    da = np.append(da, dao, axis =1)
    da = np.append(da, dag, axis=1)

    db = np.sum(da, axis=0)
    dx = np.dot(da, Wx.T)
    dWx = np.dot(x.T, da)
    dWh = np.dot(prev_h.T, da)
    dprev_h = np.dot(da, Wh.T)

    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    N, T, D = x.shape
    H = h0.shape[1]
    h = np.zeros((N, T, H))
    next_h = h0
    cache = list()
    next_c = np.zeros((N,H))
    for i in range(T):
        cur_x = x[:, i, :]
        next_h,  next_c, cur_cache = lstm_step_forward(cur_x, next_h,next_c, Wx, Wh, b)
        h[:, i, :] = next_h
        cache.append(cur_cache)
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    x, prev_h, prev_c, next_c, next_h, Wx, Wh, b, i, f, o, g, ai, af, ao, ag = cache[0]
    N, T, H = dh.shape
    D = x.shape[1]
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros(4*H)

    cur_cache = cache[T - 1]
    dnext_h = dh[:, T - 1, :]
    dnext_c = np.zeros((prev_c.shape))

    dx_cur, cur_dprev_h1, dprev_c1, dWx_cur, dWh_cur, db_cur = lstm_step_backward(dnext_h, dnext_c, cur_cache)

    dx[:, T - 1, :] = dx_cur
    dWx += dWx_cur
    dWh += dWh_cur
    db += db_cur
    dh0 += cur_dprev_h1

    for i in range(T - 2, -1, -1):
        cur_cache = cache[i]
        dx_cur, cur_dprev_h1, dprev_c1, dWx_cur, dWh_cur, db_cur = lstm_step_backward(cur_dprev_h1,dprev_c1, cur_cache)
        dx[:, i, :] += dx_cur
        dWx += dWx_cur
        dWh += dWh_cur
        db += db_cur

        dnext_h = dh[:, i, :]
        cur_dx, cur_dprev_h2, dprev_c2, cur_dWx, cur_dWh, cur_db = lstm_step_backward(dnext_h,dnext_c, cur_cache)
        dx[:, i, :] += cur_dx
        dWx += cur_dWx
        dWh += cur_dWh
        db += cur_db

        dprev_c1 += dprev_c2
        cur_dprev_h1 += cur_dprev_h2
    dh0 = cur_dprev_h1
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
