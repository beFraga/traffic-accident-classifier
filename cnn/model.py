import torch

import time
import pickle
import numpy as np
from tqdm import tqdm

from cnn.network import AccidentClassifierNet, TransferResnet
from cnn.losses import AccidentDetectionLoss
from cnn.early_stopping import EarlyStopping

class BaseModel:
    def __init__(self, save_dir, dataset, parameters, device=None, state_dict="trained_net_state_dict.pt", save_ckpt=False, early_stopping=False):
        self.state_dict = state_dict
        self.history_file = "history.pkl"
        
        self.save_dir = save_dir
        if not self.save_dir.is_dir():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        assert self.save_dir.is_dir()

        self._save_ckpt = save_ckpt

        self.start_time = time.time()

        self.start_epoch = 0
        self.cur_epoch = 0

        self.params = parameters
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.max_epochs = parameters["max_epochs"]

        self.train_dataset, self.val_dataset, self.test_dataset = dataset.get_loaders()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if early_stopping:
            patience = self.params.get("patience_epochs", 10)
            self.early_stopping = EarlyStopping(patience, min_delta=1e-4)
        else:    
            self.early_stopping = None
        self.schedulers = []

        self.best_val_loss = float("inf")

        self.history = {}

    def train_one_epoch(self):
        raise NotImplementedError()

    def validate_training(self):
        raise NotImplementedError()

    def run_test(self):
        raise NotImplementedError()

    def _append_history(self, metrics, prefix):
        for key, value in metrics.items():

            hist_key = f"{prefix}_{key}"

            if hist_key not in self.history:
                self.history[hist_key] = []

            self.history[hist_key].append(float(value))

    def train(self):
        _div = len(self.train_dataset) / self.batch_size
        _remain = int(len(self.train_dataset) % self.batch_size > 0)
        num_it_per_epoch = _div + _remain

        for e in tqdm(range(self.start_epoch, self.max_epochs)):
            self.cur_epoch = e

            self.net.train()
            train_loss = self.train_one_epoch()

            self.net.eval()
            with torch.no_grad():
                val_loss = self.validate_training()

            self._append_history(train_loss, "train_loss")
            self._append_history(val_loss, "validation_loss")

            if self.schedulers:
                for sche in self.schedulers:
                    sche.step()

            if val_loss["total"] < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss["total"]
                self.save_network()

            if self.early_stopping is not None:
                if self.early_stopping.step(val_loss):
                    print("Early stopping triggered.")
                    break
            
            if self._save_ckpt:
                save_freq = max(1, self.max_epochs // 4)
                if (save_freq == 0) or (e == self.max_epochs - 1):
                    ckpt_path = self.save_dir / f"ckpt_epoch{str(e + 1).zfill(3)}.tar"
                    self.save_model_ckpt(ckpt_path, e)

        self.history["elapsed"] = time.time() - self.start_time
        self.save_history()

    def save_network(self):
        torch.save(self.net.state_dict(), self.save_dir / self.state_dict)

    def load_network(self):
        self.net.load_state_dict(torch.load(self.save_dir / self.state_dict, map_location=self.device))

    def save_history(self):
        with open(self.save_dir / self.history_file, "wb") as fp:
            pickle.dump(self.history, fp)

    def print_history(self):
        if not self.history:
            print("There is no training history")
            return

        if "elapsed" in self.history:
            elapsed = self.history["elapsed"]
            print(f"Elapsed time: {elapsed:.2f} seconds")
        
    def save_model_ckpt(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }, path)

    def restore_model_ckpt(self, ckpt_file):
        ckpt = torch.load(ckpt_file)
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.cur_epoch = self.start_epoch
        self.history = ckpt.get("history", self.history)


class AccidentClassifier(BaseModel):
    def __init__(self, save_dir, dataset, parameters, device=None, state_dict="accidentclassifier_state_dict.pt", save_ckpt=False, early_stopping=False, tp=0):
        super().__init__(save_dir, dataset, parameters, device=device, state_dict=state_dict, save_ckpt=save_ckpt, early_stopping=early_stopping)

        self.history_file = "accidentclassifier_history.pkl"

        num_classes = self.params.get("num_classes", 5)
        S = self.params.get("S", 7)

        if tp == 1:
            self.net = TransferResnet(num_classes=num_classes, S=S)
        else:
            self.net = AccidentClassifierNet(num_classes=num_classes, S=S)
        self.net.to(self.device)

        self.loss = AccidentDetectionLoss(num_classes=num_classes)

        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(), lr=self.learning_rate
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            parameters["lr_decay_every_n_epoch"],
            gamma=parameters["lr_decay_rate"],
        )
        lr_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=15,
            T_mult=2,
            eta_min=1e-6
        )
        self.schedulers = [lr_scheduler, lr_cosine]

    def train_one_epoch(self):
        loss_numerics = {key: 0.0 for key in self.loss.key_names}
        count_loop = 0

        for images, targets in self.train_dataset:
            count_loop += 1
            self.optimizer.zero_grad()

            images = images.to(self.device)
            targets = targets.to(self.device)
            
            if np.random.rand() > 0.5:
                lam = np.random.beta(0.2, 0.2)
                batch_size = images.size()[0]
                index = torch.randperm(batch_size).to(self.device)

                images = lam * images + (1 - lam) * images[index, :]
                targets = lam * targets + (1 - lam) * targets[index, :]

            predictions = self.net(images)

            loss = self.loss(predictions, targets)
            loss["total"].backward()
            self.optimizer.step()

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        return {key: loss_numerics[key] / count_loop for key in self.loss.key_names}

    def validate_training(self):
        loss_numerics = {key: 0.0 for key in self.loss.key_names}
        count_loop = 0

        for images, targets in self.val_dataset:
            count_loop += 1
            
            images = images.to(self.device)
            targets = targets.to(self.device)

            predictions = self.net(images)

            loss = self.loss(predictions, targets)

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        return {key: loss_numerics[key] / count_loop for key in self.loss.key_names}
    
    def process(self):
        raise NotImplementedError()
