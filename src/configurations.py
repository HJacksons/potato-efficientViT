import os
import torch
import torch.optim as optim
import torch.nn as nn
from models import (ViT, EfficientNetV2B3, EfficientNetV2B3ViT, MobileNetV3_large, VGG16, ResNet50, DenseNet121,
                    MobileNetV3ViT, VGG16ViT, ResNet50ViT, DenseNet121ViT
                    )
from time import gmtime, strftime
from dotenv import load_dotenv
import wandb
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
load_dotenv()
time = strftime("%d- %H:%M:%S", gmtime())

# Dataset configurations
DATA = "../data/potatodata"  # "../data/plantVillage"
TEST_SIZE = 0.1
VALI_SIZE = 0.1
RANDOM_STATE = 42  # for reproducibility
BATCH_SIZE = 64
CLASSES = sorted(os.listdir(DATA))

# Training configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.CrossEntropyLoss()
EPOCHS = 70
lr = 0.0001

# Switch between training and testing, augmenting the dataset, and the type of dataset
TRAINING = True
AUGMENT = True  # Always true to improve model generalization
DATATYPE = "potatodata"  # plantVillage or potatodata

NEW_DATASET = True  # for the purpose of testing

if TRAINING:
    MODELS = {
        "EffNetViT": EfficientNetV2B3ViT().to(DEVICE),
        # Can add more models here
    }

    OPTIMIZERS = {
        "EffNetViT": optim.Adam(MODELS["EffNetViT"].parameters(), lr, weight_decay=0.0001),
    }
    SCHEDULER = {
        "EffNetViT": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["EffNetViT"], patience=5, factor=0.5, verbose=True
        ),
    }


else:  # Testing
    MODELS = {
        "EffNetV2B3ViT": EfficientNetV2B3ViT,
    }

SAVED_MODELS = {}
if NEW_DATASET:
    if AUGMENT:
        SAVED_MODELS = {
            "EffNetViT": "EfficientNetV2B3ViT_potatodata_AMobileug_True_014437_CNNs.pth",
        }
    else:
        # For NEW_DATASET and not AUGMENT
        SAVED_MODELS = {}
else:
    # Not NEW_DATASET
    if AUGMENT:
        SAVED_MODELS = {
            "EffNetViT": "EfficientNetV2B3_last_plantVillage_Aug_True_151547_HT400k.pth",
        }
    else:
        # Not NEW_DATASET and not AUGMENT
        SAVED_MODELS = {}

# logging results to wandb
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    name=f"{time}_{DATATYPE}_train_Aug_{AUGMENT}",  # Train/Test name
)
