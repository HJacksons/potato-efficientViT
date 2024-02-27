import torch
import torch.optim as optim
import torch.nn as nn
from models import ViT, EfficientNetV2B3, HybridModel
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
CRITERION = nn.CrossEntropyLoss()
EPOCHS = 50
lr = 0.0001

DATA = "../data/plantVillage"
TEST_SIZE = 0.2
VALI_SIZE = 0.5
RANDOM_STATE = 42  # for reproducibility
BATCH_SIZE = 64
CLASSES = sorted(os.listdir(DATA))
# print list of classes
# for i, cls in enumerate(CLASSES):
#     print(f"{i}: {cls}")

TRAINING = True
AUGMENT = True

NEW_DATASET = False  # for the purpose of testing

if TRAINING:
    MODELS = {
        "EfficientNetV2B3": EfficientNetV2B3().to(DEVICE),
        "ViT": ViT().to(DEVICE),
        "HybridModel": HybridModel().to(DEVICE),
    }

    OPTIMIZERS = {
        "EfficientNetV2B3": optim.Adam(MODELS["EfficientNetV2B3"].parameters(), lr),
        "ViT": optim.Adam(MODELS["ViT"].parameters(), lr),
        "HybridModel": optim.Adam(MODELS["HybridModel"].parameters(), lr),
    }
else:  # Testing
    MODELS = {
        "EfficientNetV2B3": EfficientNetV2B3,
        "ViT": ViT,
        "HybridModel": HybridModel,
    }

if NEW_DATASET:
    if AUGMENT:
        SAVED_MODELS = {
            "ViT": "ViT_best_model_Aug_True_153455.pth",
        }
    else:
        SAVED_MODELS = {
            "ViT": "ViT_best_model_Aug_False_153348.pth",
        }
else:
    if AUGMENT:
        SAVED_MODELS = {
            "ViT": "ViT_best_plantds_Aug_True_192710.pth",
        }
    else:
        SAVED_MODELS = {
            "ViT": "ViT_best_plantds_Aug_False_185343.pth",
        }

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    name=f"{time}_p_train_Aug_{AUGMENT}_{EPOCHS}epochs_bsize_{BATCH_SIZE}",  # Train name
    # name=f"{time}_plantD_test_models_Aug_{AUGMENT}_bsize_{BATCH_SIZE}",  # Test name
)
