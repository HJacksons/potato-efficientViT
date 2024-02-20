import torch
import torch.optim as optim
import torch.nn as nn
from models import VGG19, ResNet50, MobileNetV2
import os
import wandb
from time import gmtime, strftime

# print time now
time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

# from dataset import Dataset
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "VGG19": VGG19().to(DEVICE),
    "ResNet50": ResNet50().to(DEVICE),
    "MobileV2": MobileNetV2().to(DEVICE),
    # "model4": ModelD().to(DEVICE)
}

OPTIMIZERS = {
    "VGG19": optim.Adam(MODELS["VGG19"].parameters(), lr=0.001),
    "ResNet50": optim.Adam(MODELS["ResNet50"].parameters(), lr=0.001),
    "MobileV2": optim.Adam(MODELS["MobileV2"].parameters(), lr=0.001),
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
AUGMENT = False


# wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    name=f"{time}_Train_Aug_{AUGMENT}_{EPOCHS}epochs_batch_size_{BATCH_SIZE}",
)
