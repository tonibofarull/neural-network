import numpy as np

"""
If forward -> compute activation function from Z
Else -> compute derivative (from Z or A)
"""

eps = 1e-8

def sigmoid(Z, A=None, forward=True):
    if forward:
        return 1/(1+np.exp(-Z))
    return A*(1-A)

def tanh(Z, A=None, forward=True):
    if forward:
        a = np.exp(Z)
        b = np.exp(-Z)
        return np.divide(a-b, a+b + eps)
    return 1-A*A

def relu(Z, A=None, forward=True):
    if forward:
        return np.maximum(0,Z)
    a = np.zeros((Z.shape[0],Z.shape[1]))
    a[Z > 0] = 1
    return a
