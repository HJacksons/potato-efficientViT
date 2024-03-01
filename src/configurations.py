import torch
import torch.optim as optim
import torch.nn as nn
from models import ViT, EfficientNetV2B3, HybridModel
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
EPOCHS = 70  # From 50 to 70 for vit to learn more
lr = 0.0001

DATA = "../data/plantVillage"
TEST_SIZE = 0.1
VALI_SIZE = 0.1
RANDOM_STATE = 42  # for reproducibility
BATCH_SIZE = 64
CLASSES = sorted(os.listdir(DATA))
# print list of classes
# for i, cls in enumerate(CLASSES):
#     print(f"{i}: {cls}")

TRAINING = False
AUGMENT = True
DATATYPE = "plantVillage"  # plantVillage or potatodata

NEW_DATASET = False  # for the purpose of testing

if TRAINING:
    MODELS = {
        # "EfficientNetV2B3": EfficientNetV2B3().to(DEVICE),
        "ViT": ViT().to(DEVICE),
        "HybridModel": HybridModel().to(DEVICE),
    }
    model = MODELS["HybridModel"]  # Your hybrid model instance

    OPTIMIZERS = {
        # "EfficientNetV2B3": optim.Adam(MODELS["EfficientNetV2B3"].parameters(), lr),
        "ViT": optim.Adam(MODELS["ViT"].parameters(), lr),  #  swtich to AdamW all
        "HybridModel": optim.Adam(
            MODELS["HybridModel"].parameters(), lr, weight_decay=0.5
        ),
    }
    SCHEDULER = {
        "ViT": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["ViT"], patience=5, factor=0.5, verbose=True
        ),
        "HybridModel": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["HybridModel"], patience=2, factor=0.5, verbose=True
        ),
    }


else:  # Testing
    MODELS = {
        # "EfficientNetV2B3": EfficientNetV2B3
        "ViT": ViT,
        "HybridModel": HybridModel,
    }

if NEW_DATASET:
    if AUGMENT:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_potatodata_Aug_True_015623.pth",
            "ViT": "ViT_last_potatodata_Aug_True_134241_L2_dropout_hybrid.pth",
            "HybridModel": "HybridModel_last_potatodata_Aug_True_134241_L2_dropout_hybrid.pth",
        }
    else:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_potatodata_Aug_False_082520.pth",
            "ViT": "ViT_last_potatodata_Aug_False_134753_L2_dropout_hybrid.pth",
            "HybridModel": "HybridModel_last_potatodata_Aug_False_134753_L2_dropout_hybrid.pth",
        }
else:
    if AUGMENT:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_plantVillage_Aug_True_142228.pth",
            "ViT": "ViT_last_plantVillage_Aug_True_223713_L2_dropout_hybrid.pth",
            "HybridModel": "HybridModel_last_plantVillage_Aug_True_223713_L2_dropout_hybrid.pth",
        }
    else:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_plantVillage_Aug_False_141359.pth",
            "ViT": "ViT_last_plantVillage_Aug_False_214638_L2_dropout_hybrid.pth",
            "HybridModel": "HybridModel_last_plantVillage_Aug_False_214638_L2_dropout_hybrid.pth",
        }

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    # name=f"{time}_{DATATYPE}_train_Aug_{AUGMENT}_Vit_Hybrid_l2_4",  # Train name # Added L2 regularization... 0.5
    name=f"{time}_{DATATYPE}_test_Aug_{AUGMENT}_Vit_and_hybrid_",  # Test names
)
