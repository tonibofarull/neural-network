import numpy as np

def sigmoid(ZA, forward=True):
    """
    If forward -> compute sigmoid from Z
    Else -> compute derivate from A = g(Z)
    """
    if forward:
        return 1/(1+np.exp(-ZA))
    return ZA*(1-ZA)

def tanh(ZA, forward=True):
    """
    If forward -> compute sigmoid from Z
    Else -> compute derivate from A = g(Z)
    """
    if forward:
        a = np.exp(ZA)
        b = np.exp(-ZA)
        return np.divide(a-b, a+b + 1e-8)
    return 1-ZA*ZA
