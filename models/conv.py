from cnn.dataset import _
from cnn.model import _

import torch
import sys
import time
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORKDIR = Path().absolute()
DATASET_PATH = WORKDIR / "datasets" / "InfraredSolarModules"
META_PATH = DATASET_PATH / "module_metadata.json"
IMAGES_PATH = DATASET_PATH / "images"

print("Work directory: %s" % WORKDIR)
print("Dataset directory: %s" % DATASET_PATH)

with open("./parameters.yaml", "r") as yf:
    parameters = yaml.load(yf, Loader=yaml.SafeLoader)

params = parameters["_"]

SAVE_DIR = WORKDIR / "training"


def train():
    start_time = time.time()
    print("----- Starting CNN Train -----")
    print("----- Generating Infrared Dataset -----")
    dataset = _(META_PATH, IMAGES_PATH, train_sample=params["train_sample"], batch_size=params['batch_size'])

    print(f"Generated {len(dataset)} samples")
    model = _(SAVE_DIR, dataset, params, device=device)
    model.train()

    print("Total training time (CNN):")
    print(time.time() - start_time)
    print("Loss total (CNN):")
    print(model.history["train_loss_total"])
    run(dataset=dataset)



def run(dataset=None):
    print("----- Starting DualTaskAE Test -----")
    if dataset is None:
        print("----- Generating Seismic Noise Dataset -----")
        dataset =  _(META_PATH, IMAGES_PATH, train_sample=params["train_sample"], batch_size=params['batch_size'])

    print(f"Generated {len(dataset)} samples")
    model = _(SAVE_DIR, dataset, params)
    model.load_network()
    results = model.run_test()


switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    sys.exit(switch[sys.argv[1]]())