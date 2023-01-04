import logging
from flearn.client import Client
import math
import numpy as np

class Server(object):
    def __init__(self, data, model, parsed):
        logging.info('Using Federated avg to Train')
        self.data = data
        for key, val in parsed.items():
            logging.info("%s: %s" % (key, val))
            setattr(self, key, val)

        # initialize client
        self.clients = self.setup_clients(data, model)
        self.latest_model = None

    def get_epoch_learning_rates(self, i, initial_learningrate, maxlr, num_epochs):
        learning_rates = []
        for j in range(num_epochs):
            lr = initial_learningrate * self.total_samples / ((i * num_epochs + j)  + 1)  # cn/(1 + t)
            lr = min(maxlr, lr)     # min(32, cn/1+t)
            learning_rates.append(lr)
        return learning_rates

    def train(self, results_filename):
        # learning_rate = self.learning_rate
        if self.is_decay:
            learning_rates = self.get_epoch_learning_rates(0, self.learning_rate, self.lrconst, self.num_epochs)
        else:
            learning_rates = [self.learning_rate * self.lrconst] * self.num_epochs
        train_loss, test_loss = 0, 0
        final_iteration = 0
        is_break = False
        train_losss, test_losss, train_accs, test_accs = [], [], [], []
        num_selected_clients = max(int(self.partial_participation * len(self.clients)), 1)
        for i in range(self.num_rounds):
            final_iteration = i * self.num_epochs
            if i % self.eval_every == 0:
                train_loss, test_loss, train_acc, test_acc = self.evaluation()
                train_losss.append(train_loss); test_losss.append(test_loss)
                train_accs.append(train_acc); test_accs.append(test_acc)
                logging.info("Round {} Train_loss {}, Test_loss {}, Train_acc {}, Test_acc {}, lr {}".format(i,
                                                                                                      train_loss,
                                                                                                      test_loss,
                                                                                                      train_acc,
                                                                                                      test_acc, learning_rates[-1]))
                if train_loss < self.epsilon and self.model == 'linearregression':
                    logging.info("Final iterations {}".format(final_iteration))
                    is_break = True
                    break
                if train_acc > self.epsilon and self.model == 'logisticregression':
                    logging.info("Final iterations {}".format(final_iteration))
                    is_break = True
                    break
                if train_loss < self.epsilon and self.model == 'binarylogisticregression':
                    logging.info("Final iterations {}".format(final_iteration))
                    is_break = True
                    break
                if math.isnan(train_loss) or math.isinf(train_loss):
                    logging.info("Final iterations {}".format(1e8))
                    is_break = True
                    break

            if final_iteration > self.num_iterations:
                break

            active_clients = self.selected_clients(i,
                                                   num_selected_clients) if self.partial_participation != 1 else self.clients

            csolns = []  # buffer for receiving client solutions
            for client in active_clients:
                # client.set_params(self.latest_model)
                self.update_local_model(client, learning_rates)
                csolns.append(client.get_params())
            self.latest_model = self.aggregate(csolns)
            self.sync_clients(self.latest_model)

            # warmup = 100
            if self.is_decay:
                learning_rates = self.get_epoch_learning_rates(i, self.learning_rate, self.lrconst, self.num_epochs)
                # learning_rate = 1/(i * self.learning_rate + self.lrconst)  # see ICLR'20 Xiang Li, C.5

            if (i * self.num_epochs) % 100 == 0:
                results = [train_losss, test_losss, train_accs, test_accs, final_iteration]
                np.save(results_filename, results)

        if is_break is False:
            logging.info("Final iterations {}".format(final_iteration))

        results = [train_losss, test_losss, train_accs, test_accs, final_iteration]
        np.save(results_filename, results)

        return results

    def update_local_model(self, client, learning_rates):
        client.localsgd(self.num_epochs, learning_rates, self.batch_size, self.adapt)

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        # model = model(self.dimension)
        if self.model == 'linearregression':
            all_clients = [Client(u, self.seed, train_data[u], test_data[u], model(self.dimension)) for u in users]
        elif self.model in ['logisticregression', 'binarylogisticregression']:
            all_clients = [Client(u, self.seed, train_data[u], test_data[u], model(self.dimension, self.regularization)) for u in users]
        else:
            raise NameError('No such model')
        self.num_samples = [len(train_data[u]['x']) for u in users]
        self.test_num_samples = [len(test_data[u]['x']) for u in users]
        self.total_samples = sum(self.num_samples)
        self.test_samples = sum(self.test_num_samples)
        return all_clients

    def aggregate(self, csolns):
        latest_model = csolns[0]
        for item in csolns[1:]:
            latest_model += item
        return latest_model/len(csolns)

    def sync_clients(self, latest_model):
        for client in self.clients:
            client.set_params(latest_model)

    def evaluation(self):
        train_loss, test_loss = 0, 0
        # train_losses, test_losses = [], []
        train_acc, test_acc = 0, 0
        for i, client in enumerate(self.clients):
            # train_losses.append(client.compute_train_loss())
            # test_losses.append(client.compute_test_loss())
            train_loss += client.compute_train_loss() * self.num_samples[i]
            test_loss += client.compute_test_loss() * self.test_num_samples[i]

            if self.model in ['logisticregression', 'binarylogisticregression']:
                train_acc += client.compute_train_accuracy() * self.num_samples[i]
                test_acc  += client.compute_test_accuracy() * self.test_num_samples[i]
        train_loss, test_loss = train_loss / self.total_samples, test_loss / self.test_samples
        train_acc,  test_acc  = train_acc / self.total_samples,  test_acc / self.test_samples
        return train_loss, test_loss, train_acc, test_acc

    def selected_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients

        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return [self.clients[idx] for idx in indices]