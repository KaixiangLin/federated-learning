import json, math, os, sys, errno
import numpy as np
import random
from tqdm import trange

NUM_USER = 10

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def generate_synthetic(W, b, seed, num_samples, dimension):

    num_samples = num_samples
    samples_per_user = NUM_USER * [int(num_samples / NUM_USER)]
    # assert num_samples % NUM_USER == 0     # If false, number of samples changed.
    print(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    #### define some eprior ####
    # mean_x = np.zeros((NUM_USER, dimension))
    mean_x = np.zeros((1, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    np.random.seed(seed)
    X = np.random.multivariate_normal(mean_x[0], cov_x, num_samples)
    for i in range(NUM_USER):
        xx = X[i * samples_per_user[i]: (i + 1) * samples_per_user[i]]
        # yy = np.zeros(samples_per_user[i])
        # for j in range(samples_per_user[i]):
        yy = np.dot(xx, W) + b

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} exampls".format(i, len(y_split[i])))
    #     print(np.sum(X_split[i]), np.sum(y_split[i]))
    print("++++++++ sum of X ++++++++", np.sum(X))
    return X_split, y_split


def generate_solution(seed, dimension, NUM_CLASS):
    np.random.seed(seed)
    W = np.random.normal(0, 1, (dimension, NUM_CLASS))
    b = np.random.normal(0, 1, NUM_CLASS)
    return W, b

def run():
    # train_path = "data/train/mytrain.json"
    # test_path = "data/test/mytest.json"
    seed = 1025  #default seed 1024
    # path = "/media/Research/linkaixi/AllData/federatedlearning/synthetic_linear_regression/"
    # path = "/media/Research/linkaixi/AllData/federatedlearning/synthetic_linear_regression_1k_6k/"
    path = "/mnt/research/illidan/linkaixi/AllData/federatedlearning/synthetic_linear_regression_1k_6k_{}/".format(seed)
    # path = "/mnt/research/illidan/linkaixi/AllData/federatedlearning/"
    train_path = path + "data/train_{}/mytrain.json".format(NUM_USER)
    test_path = path + "data/test_{}/mytest.json".format(NUM_USER)

    if not os.path.exists(path + "data/test_{}".format(NUM_USER)):
        mkdir_p(path + "data/test_{}".format(NUM_USER))
    if not os.path.exists(path + "data/train_{}".format(NUM_USER)):
        mkdir_p(path + "data/train_{}".format(NUM_USER))
    # d = 300, n = 49749
    # dimension = 300
    dimension = 1000
    NUM_CLASS = 1
    W, b = generate_solution(seed, dimension, NUM_CLASS)

    X_train, y_train = generate_synthetic(W, b, seed=0+seed, num_samples=6000, dimension=dimension)
    X_test, y_test = generate_synthetic(W, b,  seed=1+seed, num_samples=1000, dimension=dimension)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': [], 'model': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    error = 0
    for i in trange(NUM_USER, ncols=120):
        uname = 'f_{0:05d}'.format(i)
        # combined = list(zip(X[i], y[i]))
        # random.shuffle(combined)
        # X[i][:], y[i][:] = zip(*combined)
        # num_samples = len(X[i])
        # train_len = int(0.9 * num_samples)
        # test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X_train[i], 'y': y_train[i]}
        train_data['num_samples'].append(len(X_train))
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X_test[i], 'y': y_test[i]}
        test_data['num_samples'].append(len(X_test))
        error += np.sum((np.dot(np.array(X_test[i]), W) + b - np.array(y_test[i])) ** 2)

    train_data['model'] = [W.tolist(), b.tolist()]
    # compute optimal loss value. verify
    print(error)
    # ++++++++ sum of X ++++++++ 18.057827259223103
    # ++++++++ sum of X ++++++++ 146.8384380986664
    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    # NUM_USER = int(sys.argv[1])
    # 10, 20, 40, 50, 100, 500, 600, 750,
    for i in [10, 20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 750, 1000]:
        NUM_USER = i
        run()

