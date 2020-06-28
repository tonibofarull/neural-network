import numpy as np
from utils import sigmoid, tanh
import matplotlib.pyplot as plt

np.random.seed(44)

class Layer():
    
    learning_rate = 0.01
    reduce_value = 0.01
    lambd = 0.01
    eps = 1e-8
    beta1 = 0.9 # Momentum
    beta2 = 0.999 # RMSprop

    def __init__(self, inputs, outputs, activation_func=sigmoid):
        self.activation_func = activation_func
        self.W = np.random.randn(outputs, inputs) * Layer.reduce_value
        self.b = np.zeros((outputs, 1))
        self.A = None # g(W*X+b)

        self.vdW = np.zeros((outputs, inputs))
        self.sdW = np.zeros((outputs, inputs))
        self.vdb = np.zeros((outputs, 1))
        self.sdb = np.zeros((outputs, 1))

    def forward_propagation(self, X):
        self.m = X.shape[1]

        self.X = X
        self.Z = np.dot(self.W, X) + self.b
        self.A = self.activation_func(self.Z)
        return self.A

    def back_propagation(self, dA, t):
        dZ = dA * self.activation_func(self.A, forward=False)
        # Gradient Descent + L2-regularization
        dW = 1/self.m * np.dot(dZ, self.X.T) + Layer.lambd/self.m * self.W
        db = 1/self.m * np.sum(dZ)

        # ADAM optimization with Bias correction
        self.vdW = Layer.beta1 * self.vdW + (1-Layer.beta1) * dW
        self.sdW = Layer.beta2 * self.sdW + (1-Layer.beta2) * np.square(dW)
        self.vdb = Layer.beta1 * self.vdb + (1-Layer.beta1) * db
        self.sdb = Layer.beta2 * self.sdb + (1-Layer.beta2) * np.square(db)
        vdW = np.divide(self.vdW, 1-Layer.beta1**t)
        sdW = np.divide(self.sdW, 1-Layer.beta2**t)
        vdb = np.divide(self.vdb, 1-Layer.beta1**t)
        sdb = np.divide(self.sdb, 1-Layer.beta2**t)
        act_W = np.divide(vdW, np.sqrt(sdW + Layer.eps))
        act_b = np.divide(vdb, np.sqrt(sdb + Layer.eps))

        self.W = self.W - Layer.learning_rate * act_W
        self.b = self.b - Layer.learning_rate * act_b

        dA = np.dot(self.W.T, dZ)
        return dA

    def get_A(self):
        return self.A

    def get_norm2(self):
        return np.linalg.norm(self.W, ord=2)

class NN():

    def __init__(self, dims):
        self.L = len(dims)
        self.layers = []
        self.costs = []
        self.initialize_layers(dims)

    def initialize_layers(self, dims):
        for i in range(1,self.L):
            layer = None
            if i == self.L-1:
                layer = Layer(dims[i-1], dims[i], activation_func=sigmoid)
            else:
                layer = Layer(dims[i-1], dims[i], activation_func=tanh)
            self.layers.append(layer)

    def predict(self, X):
        self._feed_forward(X)
        return self.layers[-1].get_A()

    def train(self, X, Y, iters=100001, t=0):
        self.m = Y.shape[1]

        for i in range(1,iters):
            i = i + t*iters
            A = self._feed_forward(X)
            cost, dA = self._cost(Y, A)
            self._back_propagation(dA, i)
            self.costs.append(cost)
            print(f"epoch {i}: {cost}")

    def _feed_forward(self, X):
        for i in range(self.L-1):
            X = self.layers[i].forward_propagation(X)
        return X

    def _cost(self, Y, A):
        frob = sum(self.layers[i].get_norm2() for i in range(self.L-1)) # Frobenius Norm
        cost_pos = np.multiply(Y, np.log(A + Layer.eps))
        cost_neg = np.multiply(1-Y, np.log(1-A + Layer.eps))
        cost = -1/self.m * np.sum(cost_pos + cost_neg) + Layer.lambd/(2*self.m) * frob
        dA = -np.divide(Y, A + Layer.eps) + np.divide(1-Y, 1-A + Layer.eps)
        return cost, dA

    def _back_propagation(self, dA, t):
        for i in reversed(range(self.L-1)):
            dA = self.layers[i].back_propagation(dA, t)
