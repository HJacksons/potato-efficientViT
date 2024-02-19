import torch
import torch.optim as optim
import torch.nn as nn
from models import VGG19
import os
import wandb
from dataset import Dataset
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "VGG19": VGG19().to(DEVICE),
    # "model2": ModelB().to(DEVICE),
    # "model3": ModelC().to(DEVICE),
    # "model4": ModelD().to(DEVICE)
}

OPTIMIZERS = {
    "VGG19": optim.Adam(MODELS["VGG19"].parameters(), lr=0.001),
    # "model2": optim.Adam(MODELS["modelB"].parameters(), lr=0.001),
    # "model3": optim.Adam(MODELS["modelC"].parameters(), lr=0.001),
    # "model4": optim.Adam(MODELS["modelD"].parameters(), lr=0.001)
}

CRITERION = nn.CrossEntropyLoss()
EPOCHS = 10

DATA = "../data/potatodata"
TEST_SIZE = 0.3
VALI_SIZE = 0.5
RANDOM_STATE = 42  # this is used to ensure reproducibility
BATCH_SIZE = 32
CLASSES = os.listdir(DATA)


DATASET = Dataset()
train_loader, vali_loader, test_loader = DATASET.prepare_dataset()
TRAIN_LOADER = train_loader
VALI_LOADER = vali_loader
TEST_LOADER = test_loader
