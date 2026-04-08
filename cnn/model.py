import torch

import time
import pickle
import numpy as np
from tqdm import tqdm

from cnn.network import _
from cnn.losses import _

class BaseModel:
    def __init__(self, save_dir, dataset, parameters, device=None, state_dict="trained_net_state_dict.pt"):
        self.state_dict = state_dict
        self.history_file = "history.pkl"
        self.save_dir = save_dir

        self.start_time = time.time()

        self.params = parameters
        self.start_epoch = 0
        self.cur_epoch = self.start_epoch
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.max_epochs = parameters["max_epochs"]
 
        self.train_dataset, self.val_dataset, self.test_dataset = dataset.get_loaders()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.schedulers = []

    def train_one_epoch(self):
        raise NotImplementedError()

    def validate_training(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

    def run_test(self):
        raise NotImplementedError()

    def train(self):
        _div = len(self.train_dataset) / self.batch_size
        _remain = int(len(self.train_dataset) % self.batch_size > 0)
        num_it_per_epoch = _div + _remain

        for e in tqdm(range(self.start_epoch, self.max_epochs)):
            self.train_one_epoch()
            # current_val_loss = self.validate_training()

            if self.schedulers:
                for sche in self.schedulers:
                    sche.step()

            self.cur_epoch += 1

        self.history["elapsed"] = time.time() - self.start_time
        self.save_history()
        self.save_network(self.save_dir / self.state_dict)

    def save_network(self, path):
        torch.save(self.net.state_dict(), path)

    def load_network(self):
        path = self.save_dir / self.state_dict
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def save_history(self):
        with open(self.save_dir / self.history_file, "wb") as fp:
            pickle.dump(self.history, fp)

    def print_history(self):
        if not self.history:
            print("There is no training history")
            return

        if "elapsed" in self.history:
            elapsed = self.history["elapsed"]
