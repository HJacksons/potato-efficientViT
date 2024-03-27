import torch
import torch.optim as optim
import torch.nn as nn
from models import ViT, EfficientNetV2B3, HybridModel, EfficientNetV2S, EfficientNetV2M, HybridModelV2s, HybridModelV2m
import os
import wandb
from time import gmtime, strftime
from dotenv import load_dotenv

load_dotenv()

# print time now
time = strftime("%d- %H:%M:%S", gmtime())

# from dataset import Dataset
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.CrossEntropyLoss()
EPOCHS = 70
lr = 0.0001

DATA = "../data/potatodata"  # "../data/potatodata
TEST_SIZE = 0.1
VALI_SIZE = 0.1
RANDOM_STATE = 42  # for reproducibility
BATCH_SIZE = 64
CLASSES = sorted(os.listdir(DATA))

TRAINING = False
AUGMENT = False
DATATYPE = "potatodata"  # plantVillage or potatodata

NEW_DATASET = True  # for the purpose of testing

if TRAINING:
    MODELS = {
        # "EfficientNetV2B3": EfficientNetV2B3().to(DEVICE),
        # "EfficientNetV2S": EfficientNetV2S().to(DEVICE),
        # "EfficientNetV2M": EfficientNetV2M().to(DEVICE),
        # "ViT": ViT().to(DEVICE),
        # "HybridModel": HybridModel().to(DEVICE),
        "HybridModelV2s": HybridModelV2s().to(DEVICE),
        "HybridModelV2m": HybridModelV2m().to(DEVICE),
    }
    #model = MODELS["HybridModel"]  # Your hybrid model instance

    OPTIMIZERS = {
        # "EfficientNetV2B3": optim.Adam(MODELS["EfficientNetV2B3"].parameters(), lr, weight_decay=0.0001),
        # "EfficientNetV2S": optim.Adam(MODELS["EfficientNetV2S"].parameters(), lr, weight_decay=0.0001),
        # "EfficientNetV2M": optim.Adam(MODELS["EfficientNetV2M"].parameters(), lr, weight_decay=0.0001),
        # "ViT": optim.Adam(MODELS["ViT"].parameters(), lr),  #  No weight decay for its stability
        # "HybridModel": optim.Adam(
        #     MODELS["HybridModel"].parameters(), lr, weight_decay=0.0001 # was 0.5 before
        # ),
        "HybridModelV2s": optim.Adam(MODELS["HybridModelV2s"].parameters(), lr, weight_decay=0.0001),
        "HybridModelV2m": optim.Adam(MODELS["HybridModelV2m"].parameters(), lr, weight_decay=0.0001),
    }
    SCHEDULER = {
        # "EfficientNetV2B3": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["EfficientNetV2B3"], patience=5, factor=0.5, verbose=True
        # ),
        # "EfficientNetV2S": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["EfficientNetV2S"], patience=5, factor=0.5, verbose=True
        # ),
        # "EfficientNetV2M": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["EfficientNetV2M"], patience=5, factor=0.5, verbose=True
        # ),
        # "ViT": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["ViT"], patience=5, factor=0.5, verbose=True
        # ),
        # "HybridModel": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["HybridModel"], patience=2, factor=0.5, verbose=True
        # ),
        "HybridModelV2s": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["HybridModelV2s"], patience=5, factor=0.5, verbose=True
        ),
        "HybridModelV2m": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["HybridModelV2m"], patience=5, factor=0.5, verbose=True
        ),
    }


else:  # Testing
    MODELS = {
        "EfficientNetV2B3": EfficientNetV2B3,
        "EfficientNetV2S": EfficientNetV2S,
        "EfficientNetV2M": EfficientNetV2M,
        #"ViT": ViT,
        "HybridModel": HybridModel,
    }

if NEW_DATASET:
    if AUGMENT:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_potatodata_Aug_True_015623.pth",
            #"ViT": "ViT_last_potatodata_Aug_True_134241_L2_dropout_hybrid.pth",
            #"HybridModel": "HybridModel_last_potatodata_Aug_True_134241_L2_dropout_hybrid.pth",
            #"HybridModel": "HybridModel_last_potatodata_Aug_True_204450_HT400k.pth",
        }
    else:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_potatodata_Aug_False_082520.pth",
            #"ViT": "ViT_last_potatodata_Aug_False_134753_L2_dropout_hybrid.pth",
            #"HybridModel": "HybridModel_last_potatodata_Aug_False_134753_L2_dropout_hybrid.pth",
            #"HybridModel": "HybridModel_last_potatodata_Aug_False_205609_HT400k.pth",

            "EfficientNetV2B3": "EfficientNetV2B3_potatodata_Aug_False_201907_EFF.pth",
            "EfficientNetV2S": "EfficientNetV2S_potatodata_Aug_False_215124_EFF.pth",
            "EfficientNetV2M": "EfficientNetV2M_potatodata_Aug_False_215124_EFF.pth",

        }
else:
    if AUGMENT:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_plantVillage_Aug_True_142228.pth",
            #"ViT": "ViT_last_plantVillage_Aug_True_223713_L2_dropout_hybrid.pth",
            #"HybridModel": "HybridModel_last_plantVillage_Aug_True_223713_L2_dropout_hybrid.pth",
            "HybridModel": "HybridModel_last_plantVillage_Aug_True_151547_HT400k.pth",
        }
    else:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_plantVillage_Aug_False_141359.pth",
            #"ViT": "ViT_last_plantVillage_Aug_False_214638_L2_dropout_hybrid.pth",
            #"HybridModel": "HybridModel_last_plantVillage_Aug_False_214638_L2_dropout_hybrid.pth",
            "HybridModel": "HybridModel_last_plantVillage_Aug_False_082359_HT400k.pth",
        }

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    #name=f"EFF{time}_{DATATYPE}_train_Aug_{AUGMENT}",  # Train name # Added L2 regularization... 0.5
    name=f"EFFT{time}_{DATATYPE}_test_Aug_{AUGMENT}_EFFindi",  # Test names
)
