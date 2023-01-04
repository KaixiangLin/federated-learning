import numpy as np
from scipy.special import softmax
from numpy import linalg as LA
from flearn.linearregression import Model as ModelBase

EPS = 1e-8
class Model(ModelBase):

    def __init__(self, dimension, mu):
        self.dimension = dimension
        self.num_class = num_class = 10
        self.W = np.zeros((dimension + 1, num_class))
        self.mu = mu
        # b = np.zeros((1, num_class))

    def set_params(self, model_params=None):
        '''set model parameters'''
        if model_params is not None:
            self.W = model_params

    def get_params(self):
        '''get model parameters'''
        return self.W

    def compute_prob(self, X):
        X_new = np.concatenate((X, np.ones((len(X), 1))), axis=1)
        return softmax(np.dot(X_new, self.W), axis=1)

    def compute_prediction(self, X):
        y_pred = self.compute_prob(X)
        return np.argmax(y_pred, axis=1)

    def l2reg(self):
        return self.mu * np.sum(self.W.reshape([-1]) ** 2)

    def compute_loss(self, X, y):
        m = len(X)
        y_pred = self.compute_prob(X)
        log_likelihood = -np.log(y_pred[range(m), y.astype(int)])
        loss = np.sum(log_likelihood) / m + self.l2reg()
        return loss

    def get_gradients(self, X, y):
        '''get model gradient'''
        m = len(X)
        X_new = np.concatenate((X, np.ones((m, 1))), axis=1)
        y_onehot = np.zeros((m, self.num_class))
        y_onehot[range(m), y.astype(int)] = 1
        y_pred = self.compute_prob(X)

        # err= y_pred - y_onehot
        # grad = self.mu * self.W
        # for i in range(m):
        #     grad += np.outer(err[i].T, X_new[i]).T/m
        grad = np.dot(X_new.T, (y_pred - y_onehot))/m + 2 * self.mu * self.W
        return grad

    # def grad_descent(self, X, y, learning_rate):
    #     grad = self.get_gradients(X, y)
    #     self.W = self.W - learning_rate * grad

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
# X = np.array([[1,2,3], [2,3,4]])
# y = [1, 2]
# model.compute_loss(X, y)
# model.set_params(np.ones_like(model.W))
# model.grad_check(X, y)


