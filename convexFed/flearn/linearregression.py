import numpy as np

class Model(object):
    
    def __init__(self, dimension):
        self.dimension = dimension
        self.num_class = num_class = 1
        self.W = np.zeros((dimension + 1, num_class))
        self.y_new = np.zeros((dimension + 1, num_class))
        self.y_old = np.zeros((dimension + 1, num_class))
        # b = np.zeros((1, num_class))

    def set_params(self, model_params=None):
        '''set model parameters'''
        if model_params is not None:
            self.W = model_params

    def get_params(self):
        '''get model parameters'''
        return self.W

    def compute_loss(self, X, y):
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1)
        y_pred = np.dot(X, self.W)
        loss = np.mean((y - y_pred) ** 2)
        return loss

    def get_gradients(self, X, y):
        '''get model gradient'''
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1)
        grad = 2 * (np.dot(X.T, np.dot(X, self.W) - y))/len(X)
        return grad

    def grad_descent(self, X, y, learning_rate):
        grad = self.get_gradients(X, y)
        self.W = self.W - learning_rate * grad

    def nesterov_grad_descent(self, X, y, alpha, beta):
        grad = self.get_gradients(X, y)
        self.y_new = self.W - alpha * grad
        self.W = self.y_new + beta * (self.y_new - self.y_old)
        self.y_old = self.y_new

    def grad_descent_adapt(self, X, y, learning_rate):
        grad = self.get_gradients(X, y)
        self.W = self.W - learning_rate * grad / np.linalg.norm(grad)

    def grad_check(self, X, y):
        grad = self.get_gradients(X, y)
        delta = np.random.normal(0, 1, self.W.shape)
        epsilon = 1e-8

        W1 = self.W + epsilon * delta
        W2 = self.W - epsilon * delta
        self.W = W1
        f_1 = self.compute_loss(X, y)

        self.W = W2
        f_2 = self.compute_loss(X, y)
        grad_true = (f_1 - f_2)/2
        grad_compute = np.dot(grad.T, delta*epsilon)
        print(grad_compute/grad_true)




