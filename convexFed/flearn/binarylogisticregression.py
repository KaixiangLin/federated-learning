import numpy as np
from scipy.special import softmax
from numpy import linalg as LA
from flearn.linearregression import Model as ModelBase

EPS = 1e-8

def sigmoid(x):
    "Numerically stable sigmoid function. 1/(1+exp(-z))"
    z = np.zeros_like(x)

    pos_index = np.where(x >= 0)
    neg_index = np.where(x < 0)
    z[pos_index] = 1/(1 + np.exp(-x[pos_index]))

    zn = np.exp(x[neg_index])
    z[neg_index] = zn/(1 + zn)
    return z

class Model(ModelBase):

    def __init__(self, dimension, mu):
        self.dimension = dimension
        self.num_class = num_class = 1
        self.W = np.zeros((dimension, num_class))
        self.mu = mu
        self.y_new = np.zeros((dimension, num_class))
        self.y_old = np.zeros((dimension, num_class))
        # b = np.zeros((1, num_class))

    def compute_prob(self, X, y=1):
        x = y * np.dot(X, self.W)
        return sigmoid(x)

    def compute_prediction(self, X):
        y_pred = self.compute_prob(X)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = -1
        return y_pred.reshape([-1])

    def l2reg(self):
        return self.mu * np.sum(self.W.reshape([-1]) ** 2)

    def compute_loss(self, X, y):
        m = len(X)
        # log_likelihood = np.log(1 + np.exp(-y.reshape([-1, self.num_class]) * np.dot(X, self.W)))
        z = y.reshape([-1, self.num_class]) * np.dot(X, self.W)
        log_likelihood = -np.log(sigmoid(z) + 1e-8)
        loss = np.sum(log_likelihood) / m + self.l2reg()
        return loss

    def get_gradients(self, X, y):
        '''get model gradient'''
        m = len(X)
        y = y.reshape([-1, self.num_class])
        y_pred = self.compute_prob(X, -y)
        grad = -np.sum(y * X * y_pred, axis=0).reshape([self.dimension, self.num_class])/m + 2 * self.mu * self.W
        return grad

    def grad_check(self, X, y):
        grad = self.get_gradients(X, y).reshape([-1])
        delta = np.random.normal(0, 1, self.W.shape)
        epsilons = [1e-5, 1e-6, 1e-7, 1e-8]
        W_record = self.W
        for epsilon in epsilons:
            self.W = W_record
            W1 = self.W + epsilon * delta
            W2 = self.W - epsilon * delta
            self.W = W1
            f_1 = self.compute_loss(X, y)
            self.W = W2
            f_2 = self.compute_loss(X, y)
            grad_true = (f_1 - f_2) / 2
            grad_compute = np.dot(grad.T, delta.reshape([-1]) * epsilon)
            print(epsilon, grad_compute / grad_true)


# dimension = 3
# model = Model(dimension, 1)
#
# X = np.array([[1,1,3],
#               [2,3,4]])
# y = np.array([1, -1]).reshape([-1, 1])
# print(model.compute_loss(X, y))
# model.set_params(np.ones_like(model.W))
# model.grad_check(X, y)


