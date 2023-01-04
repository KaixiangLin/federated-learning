import json, math, os, sys, errno
import numpy as np
import random
from tqdm import trange
# from scipy.special import softmax
# import cvxpy as cp

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

NUM_USER = 10

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def generate_synthetic(W, b, seed, num_samples, dimension):

    samples_per_user = NUM_USER * [int(num_samples / NUM_USER)]
    # assert num_samples % NUM_USER == 0  # If false, number of samples changed.
    print(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    #### define some eprior ####
    mean_x = np.zeros((1, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    np.random.seed(seed)
    X = np.random.multivariate_normal(mean_x[0], cov_x, num_samples)
    Y = np.zeros((num_samples, 1))
    for i in range(NUM_USER):
        xx = X[i * samples_per_user[i]: (i + 1) * samples_per_user[i]]
        tmp = np.dot(xx, W) + b
        yy = np.argmax(softmax(tmp), axis=1).reshape([-1])

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()
        Y[i * samples_per_user[i]: (i + 1) * samples_per_user[i]] = yy.reshape([-1, 1])

        print("{}-th users has {} exampls".format(i, len(y_split[i])))

    return X_split, y_split, X, Y

def generate_solution(seed, dimension, NUM_CLASS):
    np.random.seed(seed)
    W = np.random.normal(0, 1, (dimension, NUM_CLASS))
    b = np.random.normal(0, 1, NUM_CLASS)
    return W, b


def run():
    # train_path = "data/train/mytrain.json"
    # test_path = "data/test/mytest.json"
    # path = ""
    # train_path = "data/train_{}/mytrain.json".format(NUM_USER)
    # test_path = "data/test_{}/mytest.json".format(NUM_USER)
    # path = "/media/Research/linkaixi/AllData/federatedlearning/synthetic_logistic_regression_small/"
    # path = "/mnt/research/illidan/linkaixi/AllData/federatedlearning/synthetic_logistic_regression_iclr300/"
    path = "/media/Research/linkaixi/AllData/federatedlearning/synthetic_logistic_regression_iclr300/"
    train_path = path + "data/train_{}/mytrain.json".format(NUM_USER)
    test_path = path + "data/test_{}/mytest.json".format(NUM_USER)
    if not os.path.exists(path + "data/test_{}".format(NUM_USER)):
        mkdir_p(path + "data/test_{}".format(NUM_USER))
        mkdir_p(path + "data/train_{}".format(NUM_USER))

    dimension = 300
    NUM_CLASS = 10
    W, b = generate_solution(1024, dimension, NUM_CLASS)

    X_train, y_train, Xall, Yall = generate_synthetic(W, b, seed=0, num_samples=42000, dimension=dimension)
    X_test, y_test, _, _ = generate_synthetic(W, b, seed=1, num_samples=6000, dimension=dimension)

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in trange(NUM_USER, ncols=120):
        uname = 'f_{0:05d}'.format(i)
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
        train_data['num_samples'].append(len(X_train))
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test[i], 'y': y_test[i]}
        test_data['num_samples'].append(len(X_test))

    Ypred = np.argmax(softmax(np.dot(Xall, W) + b), axis=1)
    acc = np.sum(Ypred == Yall.reshape([-1])) / len(Yall)
    print("acc", acc, 'xsum', np.sum(Xall), 'ysum', np.sum(Yall))
    # acc 1.0 xsum 81.29812902076596 ysum 24283.0
    assert acc == 1

    # """cvx compute optimal solution"""
    # m = Yall.size
    # Yonehot = np.zeros((Yall.size, NUM_CLASS))
    # Yonehot[np.arange(Yall.size), Yall.reshape([-1]).astype(int)] = 1
    # w = cp.Variable((dimension + 1, 10))
    # # w.value = np.zeros((dimension + 1, 10))
    # A = np.concatenate((Xall, np.ones((m, 1))), axis=1)
    # # logexpsum = cp.sum(cp.log_sum_exp(A @ w, axis=1))
    # loss = -cp.sum(
    #     cp.multiply(Yonehot, A @ w)
    # )/m + cp.log_sum_exp(A @ w)/m
    # prob = cp.Problem(cp.Minimize(loss))
    # prob.solve()
    # print(prob.value)
    # print(prob.solution)
    # print("Is the problem DCP: ", prob.is_dcp())

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    # 10, 20, 40, 50, 100, 500, 600, 750,
    for i in [10, 20, 30, 40, 50, 100, 1000]:
        NUM_USER = i
        run()

