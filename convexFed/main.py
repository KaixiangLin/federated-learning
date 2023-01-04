import logging
import argparse
import importlib
import random
import numpy as np
import os
from utils.model_utils import read_data

DATASETS = {"synthetic_linear_regression",
            'synthetic_linear_regression_small',
            'synthetic_linear_regression_1k_6k',
            'synthetic_logistic_regression_iclr',
            'synthetic_logistic_regression_iclr300',
            'synthetic_logistic_regression_small',
            'w8a',
            'linearregressionw8a',
            "synthetic_linear_regression_noise1e-02",}

OPTIMIZERS = {"fedave", "fedmass", 'fednesterov'}
MODELS = {'linearregression', 'logisticregression', 'binarylogisticregression'}

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fednesterov')
    # default='fedavgnew')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='linearregressionw8a')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        choices=MODELS,
                        default='binarylogisticregression')  # stacked_lstm.py
    parser.add_argument('--regularization',
                        help='l2 regularization of logistic regression',
                        type=float,
                        default=1)
    # default = 'mclrnew')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=1000)
    parser.add_argument('--num_iterations',
                        help='number of iterations to simulate;',
                        type=int,
                        default=20000)  #20000

    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=4)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='initial learning rate for local sgd;',
                        type=float,
                        default=0.1)
    parser.add_argument('--adapt',
                        help='do we automatic adapt learning rate: 0: no 1:yes',
                        type=int,
                        default=0)
    parser.add_argument('--is_decay',
                        help='do we decay learning rate',
                        type=bool,
                        default=True)
    parser.add_argument('--number_user',
                        help='number of user',
                        type=int,
                        default=2)
    parser.add_argument('--dimension',
                        help='dimension of model in linear regression',
                        type=int,
                        default=300)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--partial_participation',
                        help='Full participation: 1',
                        type=float,
                        default=1)
    parser.add_argument('--epsilon',
                        help='Full participation: False',
                        type=float,
                        default=0.01)
    parser.add_argument('--lrconst',
                        help='Full participation: False',
                        type=float,
                        default=0.1)
    parser.add_argument('--machine',
                        help='running on which machine',
                        type=str,
                        default="")

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])

    model_path = '%s.%s' % ('flearn', parsed['model'])
    module = importlib.import_module(model_path)
    model = getattr(module, "Model")

    trainer_path = '%s.%s' % ('flearn', parsed['optimizer'])
    module = importlib.import_module(trainer_path)
    trainer = getattr(module, "Server")

    return parsed, model, trainer




def main():

    parsed, model, trainer = read_args()
    if parsed['machine'] == 'hpcc':
        path = '/mnt/research/illidan/linkaixi/AllData/federatedlearning/'  # on hpcc
    elif parsed['machine'] == 'gpu':
        path = '/media/Research/linkaixi/AllData/federatedlearning/'  # on gpu
    else:
        path = './data/'
        logging.error("use local data")
    if parsed['optimizer'] == "fednesterov":
        filename = '{}-{}-seed{}-epsilon{}-H{}-K{}-lr{}-decay{}-b{}-lrconst{}-regularization{}-adapt{}'.format(
            parsed['dataset'],
            parsed['optimizer'],
            parsed['seed'],
            parsed['epsilon'],
            parsed['num_epochs'],
            parsed['number_user'],
            parsed['learning_rate'],
            parsed['is_decay'],
            parsed['batch_size'],
            parsed['lrconst'],
            parsed['regularization'],
            parsed['adapt'])
    else:
        filename = '{}-seed{}-epsilon{}-H{}-K{}-lr{}-decay{}-b{}-lrconst{}-regularization{}-adapt{}'.format(
                    parsed['dataset'],
                    parsed['seed'],
                    parsed['epsilon'],
                    parsed['num_epochs'],
                    parsed['number_user'],
                    parsed['learning_rate'],
                    parsed['is_decay'],
                    parsed['batch_size'],
                    parsed['lrconst'],
                    parsed['regularization'],
                    parsed['adapt'])
    if parsed['partial_participation'] != 1:
        filename = '{}-{}-seed{}-epsilon{}-H{}-K{}-lr{}-lrconst{}-regularization{}-partial{}'.format(
            parsed['dataset'],
            parsed['optimizer'],
            parsed['seed'],
            parsed['epsilon'],
            parsed['num_epochs'],
            parsed['number_user'],
            parsed['learning_rate'],
            parsed['lrconst'],
            parsed['regularization'],
            parsed['partial_participation'])
    logname = path + 'logs/' + filename
    print("logfile", logname)
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    num_user = parsed['number_user']

    # train_path = os.path.join('data', parsed['dataset'], 'data', 'train_{}'.format(num_user))
    # test_path = os.path.join('data', parsed['dataset'], 'data', 'test_{}'.format(num_user))
    train_path = os.path.join(path, parsed['dataset'], 'data', 'train_{}'.format(num_user))
    test_path = os.path.join(path, parsed['dataset'], 'data', 'test_{}'.format(num_user))
    dataset = read_data(train_path, test_path)

    t = trainer(dataset, model, parsed)
    results_filename = path + 'results/' + filename + ".npy"
    results = t.train(results_filename)


if __name__ == "__main__":
    main()

# /media/Research/linkaixi/env/py36/bin/python /home/linkaixi/Dropbox/recsys/convexFed/main.py --dataset=synthetic_linear_regression_1k_6k --epsilon=0.01 --num_rounds=500000 --num_epochs=1 --number_user=10 --learning_rate=0.001 --is_decay=True --lrconst=1 --machine=gpu --batch_size=4 --num_iteration=50000 --model=linearregression --optimizer=fedave --dimension=1000 --adapt=1