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
TRAINING = False
AUGMENT = True  # Always true to improve model generalization
DATATYPE = "potatodata"  # plantVillage or potatodata

NEW_DATASET = True  # for the purpose of testing

if TRAINING:
    MODELS = {
        # Individual models
        "EfficientNetV2B3": EfficientNetV2B3().to(DEVICE),
        "ViT": ViT().to(DEVICE),
        "MobileNetV3_large": MobileNetV3_large().to(DEVICE),
        "VGG16": VGG16().to(DEVICE),
        "ResNet50": ResNet50().to(DEVICE),
        "DenseNet121": DenseNet121().to(DEVICE),

        # Hybrid models
        "EffNetViT": EfficientNetV2B3ViT().to(DEVICE),
        "MobileNetV3ViT": MobileNetV3ViT().to(DEVICE),
        "VGG16ViT": VGG16ViT().to(DEVICE),
        "ResNet50ViT": ResNet50ViT().to(DEVICE),
        "DenseNet121ViT": DenseNet121ViT().to(DEVICE),
    }
    # model = MODELS["HybridModel"]  # Your hybrid model instance

    OPTIMIZERS = {
        # Individual models
        "EfficientNetV2B3": optim.Adam(MODELS["EfficientNetV2B3"].parameters(), lr, weight_decay=0.0001),
        "ViT": optim.Adam(MODELS["ViT"].parameters(), lr, weight_decay=0.0001),  # Did not utelize weight decay for
        # its stability
        "MobileNetV3_large": optim.Adam(MODELS["MobileNetV3_large"].parameters(), lr, weight_decay=0.0001),
        "VGG16": optim.Adam(MODELS["VGG16"].parameters(), lr, weight_decay=0.0001),
        "ResNet50": optim.Adam(MODELS["ResNet50"].parameters(), lr, weight_decay=0.0001),
        "DenseNet121": optim.Adam(MODELS["DenseNet121"].parameters(), lr, weight_decay=0.0001),

        # Hybrid models
        "EffNetViT": optim.Adam(MODELS["EffNetViT"].parameters(), lr, weight_decay=0.0001),
        "MobileNetV3ViT": optim.Adam(MODELS["MobileNetV3ViT"].parameters(), lr, weight_decay=0.0001),
        "VGG16ViT": optim.Adam(MODELS["VGG16ViT"].parameters(), lr, weight_decay=0.0001),
        "ResNet50ViT": optim.Adam(MODELS["ResNet50ViT"].parameters(), lr, weight_decay=0.0001),
        "DenseNet121ViT": optim.Adam(MODELS["DenseNet121ViT"].parameters(), lr, weight_decay=0.0001),
    }
    SCHEDULER = {
        # Individual models
        "EfficientNetV2B3": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["EfficientNetV2B3"], patience=5, factor=0.5, verbose=True
        ),

        "ViT": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["ViT"], patience=5, factor=0.5, verbose=True
        ),

        "MobileNetV3_large": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["MobileNetV3_large"], patience=5, factor=0.5, verbose=True
        ),

        "VGG16": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["VGG16"], patience=5, factor=0.5, verbose=True
        ),
        "ResNet50": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["ResNet50"], patience=5, factor=0.5, verbose=True
        ),
        "DenseNet121": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["DenseNet121"], patience=5, factor=0.5, verbose=True
        ),

        # Hybrid models
        "EffNetViT": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["EffNetViT"], patience=5, factor=0.5, verbose=True
        ),

        "MobileNetV3ViT": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["MobileNetV3ViT"], patience=5, factor=0.5, verbose=True
        ),
        "VGG16ViT": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["VGG16ViT"], patience=5, factor=0.5, verbose=True
        ),
        "ResNet50ViT": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["ResNet50ViT"], patience=5, factor=0.5, verbose=True
        ),
        "DenseNet121ViT": optim.lr_scheduler.ReduceLROnPlateau(
            OPTIMIZERS["DenseNet121ViT"], patience=5, factor=0.5, verbose=True
        ),

    }


else:  # Testing
    MODELS = {
        # Individual models
        "EfficientNetV2B3": EfficientNetV2B3,
        "ViT": ViT,
        "MobileNetV3_large": MobileNetV3_large,
        "VGG16": VGG16,
        "ResNet50": ResNet50,
        "DenseNet121": DenseNet121,

        # Hybrid models
        "EffNetV2B3ViT": EfficientNetV2B3ViT,
        "MobileNetV3ViT": MobileNetV3ViT,
        "VGG16ViT": VGG16ViT,
        "ResNet50ViT": ResNet50ViT,
        "DenseNet121ViT": DenseNet121ViT,

    }

if NEW_DATASET:
    if AUGMENT:
        SAVED_MODELS = {
            # Individual models
            "EfficientNetV2B3": "EfficientNetV2B3_potatodata_Aug_True_013313_CNNs.pth",
            "ViT": "ViT_potatodata_Aug_True_013313_CNNs.pth",
            "MobileNetV3_large": "MobileNetV3_large_potatodata_Aug_True_013313_CNNs.pth",
            "VGG16": "VGG16_potatodata_Aug_True_013313_CNNs.pth",
            "ResNet50": "ResNet50_potatodata_Aug_True_013313_CNNs.pth",
            "DenseNet121": "DenseNet121_potatodata_Aug_True_013313_CNNs.pth",

            # Hybrid models
            "EffNetViT": "EfficientNetV2B3ViT_potatodata_AMobileug_True_014437_CNNs.pth",
            "MobileNetV3ViT": "MobileNetV3ViT_potatodata_Aug_True_014437_CNNs.pth",
            "VGG16ViT": "VGG16ViT_potatodata_AMobileug_True_014437_CNNs.pth",
            "ResNet50ViT": "ResNet50ViT_potatodata_Aug_True_014437_CNNs.pth",
            "DenseNet121ViT": "DenseNet121ViT_potatodata_Aug_True_014437_CNNs.pth",

        }
    else:
        SAVED_MODELS = {
            # Comparing model performance with and without augmentation
            # Models were loaded here  e.g.
            # "EffNetV2B3ViT": "EffNetViT_potatodata_Aug_False_015254_CNNs.pth",

        }
else:
    if AUGMENT:
        SAVED_MODELS = {
            # Individual models
            "EfficientNetV2B3": "EfficientNetV2B3_last_plantVillage_Aug_True_142228.pth",
            "ViT": "ViT_last_plantVillage_Aug_True_223713_L2_dropout_hybrid.pth",
            "MobileNetV3_large": "MobileNetV3_large_plantVillage_Aug_True_103206_CNNs.pth",
            "VGG16": "VGG16_plantVillage_Aug_True_103206_CNNs.pth",
            "ResNet50": "ResNet50_plantVillage_Aug_True_103206_CNNs.pth",
            "DenseNet121": "DenseNet121_plantVillage_Aug_True_103206_CNNs.pth",

            # Hybrid models
            "EffNetViT": "EfficientNetV2B3_last_plantVillage_Aug_True_151547_HT400k.pth",
            "MobileNetV3ViT": "MobileNetV3ViT_plantVillage_Aug_True_103206_CNNs.pth",
            "VGG16ViT": "VGG16ViT_plantVillage_Aug_True_103206_CNNs.pth",
            "ResNet50ViT": "ResNet50ViT_plantVillage_Aug_True_103206_CNNs.pth",
            "DenseNet121ViT": "DenseNet121ViT_plantVillage_Aug_True_103206_CNNs.pth",
        }
    else:
        SAVED_MODELS = {
            # Testing without augmentation
            # "DenseNet121ViT": "DenseNet121ViT_plantVillage_Aug_False_103421_CNNs.pth",
        }

# logging results to wandb
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    # name=f"{time}_{DATATYPE}_train_Aug_{AUGMENT}",  # Train name
    name=f"{time}_{DATATYPE}_test_Aug_{AUGMENT}",  # Test names
)
