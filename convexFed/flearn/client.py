import numpy as np
from utils.model_utils import batch_data

class Client(object):
    
    def __init__(self, id, seed, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, model=None):
        self.model = model
        self.seed = seed
        self.id = id # integer
        self.train_data = {k: np.array(v) for k, v in train_data.items()}
        self.eval_data = {k: np.array(v) for k, v in eval_data.items()}
        self.num_samples = len(self.train_data['y'])
        self.test_samples = len(self.eval_data['y'])

    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()

    def get_grads(self, model_len):
        '''get model gradient'''
        return self.model.get_gradients(self.train_data, model_len)

    def localsgd(self, iteration, learning_rates, batch_size, adapt=0):
        for idx in range(iteration):
            X, y = batch_data(self.train_data, batch_size, self.seed)
            # print(self.id, np.sum(X), np.sum(y),
            #       np.sum(self.train_data['x']), np.sum(self.train_data['y']))
            if adapt == 0:
                self.model.grad_descent(X, y, learning_rates[idx])
            else:
                self.model.grad_descent_adapt(X, y, learning_rates[idx])
            # self.model.grad_check(X, y)

    def nesterov_grad_descent(self, iteration, batch_size, alphas, beta):
        for idx in range(iteration):
            X, y = batch_data(self.train_data, batch_size, self.seed)
            self.model.nesterov_grad_descent(X, y, alphas[idx], beta[idx])


    def compute_train_loss(self):
        X = self.train_data['x']
        y = self.train_data['y']
        loss = self.model.compute_loss(X, y)
        return loss

    def compute_test_loss(self):
        X = self.eval_data['x']
        y = self.eval_data['y']
        loss = self.model.compute_loss(X, y)
        return loss

    def compute_train_accuracy(self):
        X = self.train_data['x']
        y = self.train_data['y']
        y_pred = self.model.compute_prediction(X)
        return np.sum(y_pred.astype(int) == y.astype(int))/len(y)

    def compute_test_accuracy(self):
        X = self.eval_data['x']
        y = self.eval_data['y']
        y_pred = self.model.compute_prediction(X)
        return np.sum(y_pred.astype(int) == y.astype(int)) / len(y)

