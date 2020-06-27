import numpy as np

def sigmoid(x, forward=True):
    """
    If forward -> compute sigmoid
    Else -> compute derivate
    """
    if forward:
        return 1/(1+np.exp(-x))
    return x*(1-x)


def relu(x, forward=True):
    """
    If forward -> compute sigmoid
    Else -> compute derivate
    """
    if forward:
        return np.maximum(0,x)
    x[x <= 0] = 0
    return x

def tanh(x, forward=True):
    if forward:
        # x = np.maximum(np.minimum(x,500), -500) # save overflow
        ex = np.exp(x)
        eminx = np.exp(-x)
        return np.divide(ex-eminx, ex+eminx + 1e-8)
    return 1-x*x