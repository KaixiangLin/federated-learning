import json, math, os, sys, errno
import numpy as np
import random
from tqdm import trange
from libsvm.svmutil import svm_read_problem
import scipy.sparse as sp

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
    return ex / sum_ex


def generate_splituser(X, Y, num_samples, W, b):
    noise_level = 0
    samples_per_user = NUM_USER * [int(num_samples / NUM_USER)]
    # assert num_samples % NUM_USER == 0  # If false, number of samples changed.
    res = num_samples % NUM_USER
    for i in range(res):
        samples_per_user[i] += 1
    print(np.sum(samples_per_user))
    assert np.sum(samples_per_user) == num_samples

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    for i in range(NUM_USER):
        xx = X[i * samples_per_user[i]: (i + 1) * samples_per_user[i]]
        if noise_level >0:
            noise = np.random.normal(0, noise_level, samples_per_user[i]).reshape((samples_per_user[i], 1))
            yy = np.dot(xx, W) + b + noise
        else:
            yy = np.dot(xx, W) + b
        Y[i * samples_per_user[i]: (i + 1) * samples_per_user[i]] = yy

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()
        print("{}-th users has {} exampls".format(i, len(y_split[i])))

    return X_split, y_split


def generate_solution(seed, dimension, NUM_CLASS):
    np.random.seed(seed)
    W = np.random.normal(0, 1, (dimension, NUM_CLASS))
    b = np.random.normal(0, 1, NUM_CLASS)
    return W, b

def load_data(train_file, dimension):
    y_train, x_train = svm_read_problem(train_file)
    N = len(x_train)  # column index 1-300

    X = np.zeros((N, dimension))
    Y = y_train
    # X = sp.dok_matrix((N, dimension))
    # Y = np.array(y_train).reshape([-1, 1])

    for row, item in enumerate(x_train):
        for k, v in item.items():
            X[row][int(k) - 1] = v

    return X, Y, N


def run():
    path = "/mnt/research/illidan/linkaixi/AllData/federatedlearning/w8a/"
    # path = "/media/Research/linkaixi/AllData/federatedlearning/linearregressionw8a/"
    # path = "./"
    train_path = path + "data/train_{}/mytrain.json".format(NUM_USER)
    test_path = path + "data/test_{}/mytest.json".format(NUM_USER)
    if not os.path.exists(path + "data/test_{}".format(NUM_USER)):
        mkdir_p(path + "data/test_{}".format(NUM_USER))
        mkdir_p(path + "data/train_{}".format(NUM_USER))

    dimension = 300
    NUM_CLASS = 1
    W, b = generate_solution(1024, dimension, NUM_CLASS)

    train_file = path + 'w8a'
    test_file = path + 'w8a.t'

    X, Y, N = load_data(train_file, dimension)
    Xall, Yall = X, Y
    X_train, Y_train = generate_splituser(X, Y, N, W, b)

    X, Y, N = load_data(test_file, dimension)
    X_test, Y_test = generate_splituser(X, Y, N, W, b)



    # X_train, y_train, Xall, Yall = generate_synthetic(W, b, seed=0, num_samples=42000, dimension=dimension)
    # X_test, y_test, _, _ = generate_synthetic(W, b, seed=1, num_samples=6000, dimension=dimension)

    # # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    error = 0
    train_error = 0
    for i in trange(NUM_USER, ncols=120):
        uname = 'f_{0:05d}'.format(i)
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_train[i], 'y': Y_train[i]}
        train_data['num_samples'].append(len(X_train[i]))
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test[i], 'y': Y_test[i]}
        test_data['num_samples'].append(len(X_test[i]))
        error += np.sum((np.dot(np.array(X_test[i]), W) + b - np.array(Y_test[i])) ** 2)
        train_error += np.sum((np.dot(np.array(X_train[i]), W) + b - np.array(Y_train[i])) ** 2)

    print("error,", error, train_error)
    #
    # Ypred = np.argmax(softmax(np.dot(Xall, W) + b), axis=1)
    # acc = np.sum(Ypred == Yall.reshape([-1])) / len(Yall)
    # print("acc", acc, 'xsum', np.sum(Xall), 'ysum', np.sum(Yall))
    # # acc 1.0 xsum 81.29812902076596 ysum 24283.0
    # assert acc == 1

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    # 10, 20, 40, 50, 100, 500, 600, 750,
    # for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        NUM_USER = i
        run()
