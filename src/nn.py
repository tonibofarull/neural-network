import numpy as np
from utils import sigmoid, tanh
import matplotlib.pyplot as plt

np.random.seed(44)

class NN():

    reduce_value = 0.01
    learning_rate = 0.01
    lambd = 0.01
    eps = 1e-8

    def __init__(self, dims):
        self.L = len(dims)
        self.Ws, self.bs = [], []
        self.vdW, self.vdb = [], []
        self.sdW, self.sdb = [], []
        self.As = None
        self.costs = []
        self.initialize_parameters(dims)

    def initialize_parameters(self, dims):
        for i in range(1,self.L):
            W = np.random.randn(dims[i], dims[i-1]) * NN.reduce_value
            b = np.zeros((dims[i], 1))
            self.Ws.append(W)
            self.bs.append(b)
            self.vdW.append(np.zeros((dims[i], dims[i-1])))
            self.sdW.append(np.zeros((dims[i], dims[i-1])))
            self.vdb.append(np.zeros((dims[i], 1)))
            self.sdb.append(np.zeros((dims[i], 1)))

    def train(self, X, Y, iters=100001, t=0, plot_cost=False):
        
        self.m = Y.shape[1]

        for i in range(1,iters):
            i = i + t*iters
            self._feed_forward(X)
            cost, dZ = self._cost(Y)
            self._back_propagation(Y, dZ, i)
            self.costs.append(cost)
            print(i, cost)
            if cost < 0.00001:
                break

    def predict(self, X):
        self._feed_forward(X)
        return self.As[-1]

    def _cost(self, Y):
        cost_pos = np.multiply(Y, np.log(self.As[-1] + NN.eps))
        cost_neg = np.multiply(1-Y, np.log(1-self.As[-1] + NN.eps))
        frob = sum(np.linalg.norm(self.Ws[i], ord=2) for i in range(len(self.Ws)))
        cost = -1/self.m * np.sum(cost_pos + cost_neg) + NN.lambd/(2*self.m) * frob
        dA = -np.divide(Y, self.As[-1] + NN.eps) + np.divide(1-Y, 1-self.As[-1] + NN.eps)
        dZ = dA * sigmoid(self.As[-1], forward=False)
        return cost, dZ

    def _feed_forward(self, X):
        self.As = [X]
        for i in range(self.L-1):
            Z = np.dot(self.Ws[i], self.As[i]) + self.bs[i]

            if i == self.L-2:
                self.As.append(sigmoid(Z))
            else:
                self.As.append(tanh(Z))

    def _back_propagation(self, Y, dZ, t):
        for i in reversed(range(self.L-1)):
            dW = 1/self.m * np.dot(dZ, self.As[i].T) + self.lambd/self.m * self.Ws[i]
            db = 1/self.m * np.sum(dZ)

            # momentum + rmsprop = adam with bias correction
            beta1 = 0.9
            beta2 = 0.999
            self.vdW[i] = beta1 * self.vdW[i] + (1-beta1) * dW
            self.sdW[i] = beta2 * self.sdW[i] + (1-beta2) * np.square(dW)
            self.vdb[i] = beta1 * self.vdb[i] + (1-beta1) * db
            self.sdb[i] = beta2 * self.sdb[i] + (1-beta2) * np.square(db)
            vdW = np.divide(self.vdW[i], 1-beta1**t)
            sdW = np.divide(self.sdW[i], 1-beta2**t)
            vdb = np.divide(self.vdb[i], 1-beta1**t)
            sdb = np.divide(self.sdb[i], 1-beta2**t)
            act_W = np.divide(vdW, np.sqrt(sdW + NN.eps))
            act_b = np.divide(vdb, np.sqrt(sdb + NN.eps))

            self.Ws[i] = self.Ws[i] - NN.learning_rate * act_W
            self.bs[i] = self.bs[i] - NN.learning_rate * act_b

            dA = np.dot(self.Ws[i].T, dZ)
            dZ = dA * tanh(self.As[i], forward=False)
