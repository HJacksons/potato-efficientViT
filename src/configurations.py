import torch
import torch.optim as optim
import torch.nn as nn
from models import VGG19, ResNet50, MobileNetV2, ViT
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

NEW_DATASET = False

if TRAINING:
    MODELS = {
        "VGG19": VGG19().to(DEVICE),
        "ResNet50": ResNet50().to(DEVICE),
        "MobileV2": MobileNetV2().to(DEVICE),
        "ViT": ViT().to(DEVICE),
    }

    OPTIMIZERS = {
        "VGG19": optim.Adam(MODELS["VGG19"].parameters(), lr),
        "ResNet50": optim.Adam(MODELS["ResNet50"].parameters(), lr),
        "MobileV2": optim.Adam(MODELS["MobileV2"].parameters(), lr),
        "ViT": optim.Adam(MODELS["ViT"].parameters(), lr),
    }
else:  # Testing
    MODELS = {
        "VGG19": VGG19,
        "ResNet50": ResNet50,
        "MobileV2": MobileNetV2,
        "ViT": ViT,
    }

if NEW_DATASET:
    if AUGMENT:
        SAVED_MODELS = {
            "VGG19": "VGG19_best_model_Aug_True_153455.pth",
            "ResNet50": "ResNet50_best_model_Aug_True_153455.pth",
            "MobileV2": "MobileV2_best_model_Aug_True_153455.pth",
            "ViT": "ViT_best_model_Aug_True_153455.pth",
        }
    else:
        SAVED_MODELS = {
            "VGG19": "VGG19_best_model_Aug_False_153348.pth",
            "ResNet50": "ResNet50_best_model_Aug_False_153348.pth",
            "MobileV2": "MobileV2_best_model_Aug_False_153348.pth",
            "ViT": "ViT_best_model_Aug_False_153348.pth",
        }
else:
    if AUGMENT:
        SAVED_MODELS = {
            # "VGG19": "VGG19_best_plantds_Aug_True_000353.pth", # 1st training
            # "ResNet50": "ResNet50_best_plantds_Aug_True_000353.pth",
            # "MobileV2": "MobileV2_best_plantds_Aug_True_000353.pth",
            # "ViT": "ViT_best_plantds_Aug_True_000353.pth",
            "VGG19": "VGG19_best_plantds_Aug_True_192710.pth",  # 2nd training
            "ResNet50": "ResNet50_best_plantds_Aug_True_192710.pth",
            "MobileV2": "MobileV2_best_plantds_Aug_True_192710.pth",
            "ViT": "ViT_best_plantds_Aug_True_192710.pth",
        }
    else:
        SAVED_MODELS = {
            # "VGG19": "VGG19_best_plantds_Aug_False_005109.pth",
            # "ResNet50": "ResNet50_best_plantds_Aug_False_005109.pth",
            # "MobileV2": "MobileV2_best_plantds_Aug_False_005109.pth",
            # "ViT": "ViT_best_plantds_Aug_False_005109.pth",
            "VGG19": "VGG19_best_plantds_Aug_False_185343.pth",
            "ResNet50": "ResNet50_best_plantds_Aug_False_185343.pth",
            "MobileV2": "MobileV2_best_plantds_Aug_False_185343.pth",
            "ViT": "ViT_best_plantds_Aug_False_185343.pth",
        }

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    name=f"{time}_plantD_train_Aug_{AUGMENT}_{EPOCHS}epochs_bsize_{BATCH_SIZE}",  # Train name
    #name=f"{time}_plantD_test_models_Aug_{AUGMENT}_bsize_{BATCH_SIZE}",  # Test name
)
