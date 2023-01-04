import logging
from flearn.client import Client
import math
import numpy as np
from flearn.fedave import Server as ServerBase

class Server(ServerBase):
    def __init__(self, data, model, parsed):
        logging.info('Using Federated avg to Train')
        self.data = data
        for key, val in parsed.items():
            logging.info("%s: %s" % (key, val))
            setattr(self, key, val)

        # initialize client
        self.clients = self.setup_clients(data, model)
        self.latest_model = None

    def update_local_model(self, client, learning_rates, betas):
        client.nesterov_grad_descent(self.num_epochs, self.batch_size, learning_rates, betas)

    def get_betas(self, rounds, num_epochs):
        # betas = [(rounds + i) / (rounds + i + 3) for i in range(num_epochs)]
        betas = [0.1] * num_epochs
        return betas

    def train(self, results_filename):
        # learning_rate = self.learning_rate
        if self.is_decay:
            learning_rates = self.get_epoch_learning_rates(0, self.learning_rate, self.lrconst, self.num_epochs)
        else:
            learning_rates = [self.learning_rate * self.lrconst] * self.num_epochs

        betas = self.get_betas(0, self.num_epochs)
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
                loginfostr = "Round {} Train_loss {}, Test_loss {}, " \
                             "Train_acc {}, Test_acc {}, lr {} beta {}".format(i,
                                                                               train_loss,
                                                                               test_loss,
                                                                               train_acc,
                                                                               test_acc,
                                                                               learning_rates[-1],
                                                                               betas[-1])
                logging.info(loginfostr)
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
                self.update_local_model(client, learning_rates, betas)
                csolns.append(client.get_params())
            self.latest_model = self.aggregate(csolns)
            self.sync_clients(self.latest_model)

            betas = self.get_betas(i, self.num_epochs)
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