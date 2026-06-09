class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        """
        patience: How many epochs to wait after last time validation loss improved.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, val_loss_dict):
        current_loss = val_loss_dict["total"]
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop