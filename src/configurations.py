import torch
import torch.optim as optim
import torch.nn as nn
from models import (ViT, EfficientNetV2B3, HybridModel, EfficientNetV2S, EfficientNetV2M,
                    HybridModelV2s, HybridModelV2m, Xception, Inceptionv3, HybridModelV2s,
                    HybridInceptionV3, HybridXception, HybridModelv2b3, MobileNetV3_large, VGG16, ResNet50, DenseNet121,
                    MobileNetV3ViT, VGG16ViT, ResNet50ViT, DenseNet121ViT
                    )
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

DATA = "../data/plantVillage"  # "../data/potatodata
TEST_SIZE = 0.1
VALI_SIZE = 0.1
RANDOM_STATE = 42  # for reproducibility
BATCH_SIZE = 64
CLASSES = sorted(os.listdir(DATA))

TRAINING = False
AUGMENT = True
DATATYPE = "plantVillage"  # plantVillage or potatodata .

NEW_DATASET = False  # for the purpose of testing

if TRAINING:
    MODELS = {
        # "EfficientNetV2B3": EfficientNetV2B3().to(DEVICE),
        # "EfficientNetV2S": EfficientNetV2S().to(DEVICE),
        # "EfficientNetV2M": EfficientNetV2M().to(DEVICE),
        # "ViT": ViT().to(DEVICE),
        # # "HybridModel": HybridModel().to(DEVICE),
        # "HybridModelV2s": HybridModelV2s().to(DEVICE),
        # "HybridModelV2m": HybridModelV2m().to(DEVICE),

        # "Xception": Xception().to(DEVICE),
        # "Inceptionv3": Inceptionv3().to(DEVICE),
        # "DenseNet121": DenseNet121().to(DEVICE),
        #"HybridInceptionV3": HybridInceptionV3().to(DEVICE),
        # "HybridXception": HybridXception().to(DEVICE),

        # "HybridModelv2b3": HybridModelv2b3().to(DEVICE),

        "MobileNetV3_large": MobileNetV3_large().to(DEVICE),
        "VGG16": VGG16().to(DEVICE),
        "ResNet50": ResNet50().to(DEVICE),
        "DenseNet121": DenseNet121().to(DEVICE),

        "MobileNetV3ViT": MobileNetV3ViT().to(DEVICE),
        "VGG16ViT": VGG16ViT().to(DEVICE),
        "ResNet50ViT": ResNet50ViT().to(DEVICE),
        "DenseNet121ViT": DenseNet121ViT().to(DEVICE),
    }
    # model = MODELS["HybridModel"]  # Your hybrid model instance

    OPTIMIZERS = {
        # "EfficientNetV2B3": optim.Adam(MODELS["EfficientNetV2B3"].parameters(), lr, weight_decay=0.0001),
        # "EfficientNetV2S": optim.Adam(MODELS["EfficientNetV2S"].parameters(), lr, weight_decay=0.0001),
        # "EfficientNetV2M": optim.Adam(MODELS["EfficientNetV2M"].parameters(), lr, weight_decay=0.0001),
        # "ViT": optim.Adam(MODELS["ViT"].parameters(), lr, weight_decay=0.0001),  # No weight decay for its stability
        # # "HybridModel": optim.Adam(
        # #     MODELS["HybridModel"].parameters(), lr, weight_decay=0.0001 # was 0.5 before
        # # ),
        # "HybridModelV2s": optim.Adam(MODELS["HybridModelV2s"].parameters(), lr, weight_decay=0.0001),
        # "HybridModelV2m": optim.Adam(MODELS["HybridModelV2m"].parameters(), lr, weight_decay=0.0001),

        # "Xception": optim.Adam(MODELS["Xception"].parameters(), lr, weight_decay=0.0001),
        # "Inceptionv3": optim.Adam(MODELS["Inceptionv3"].parameters(), lr, weight_decay=0.0001),
        # "DenseNet121": optim.Adam(MODELS["DenseNet121"].parameters(), lr, weight_decay=0.0001),
        #"HybridInceptionV3": optim.Adam(MODELS["HybridInceptionV3"].parameters(), lr, weight_decay=0.0001),
        # "HybridXception": optim.Adam(MODELS["HybridXception"].parameters(), lr, weight_decay=0.0001),

        #"HybridModelv2b3": optim.Adam(MODELS["HybridModelv2b3"].parameters(), lr, weight_decay=0.0001),

        "MobileNetV3_large": optim.Adam(MODELS["MobileNetV3_large"].parameters(), lr, weight_decay=0.0001),
        "VGG16": optim.Adam(MODELS["VGG16"].parameters(), lr, weight_decay=0.0001),
        "ResNet50": optim.Adam(MODELS["ResNet50"].parameters(), lr, weight_decay=0.0001),
        "DenseNet121": optim.Adam(MODELS["DenseNet121"].parameters(), lr, weight_decay=0.0001),

        "MobileNetV3ViT": optim.Adam(MODELS["MobileNetV3ViT"].parameters(), lr, weight_decay=0.0001),
        "VGG16ViT": optim.Adam(MODELS["VGG16ViT"].parameters(), lr, weight_decay=0.0001),
        "ResNet50ViT": optim.Adam(MODELS["ResNet50ViT"].parameters(), lr, weight_decay=0.0001),
        "DenseNet121ViT": optim.Adam(MODELS["DenseNet121ViT"].parameters(), lr, weight_decay=0.0001),
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
        # # ),
        # "ViT": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["ViT"], patience=5, factor=0.5, verbose=True
        # ),
        # # "HybridModel": optim.lr_scheduler.ReduceLROnPlateau(
        # #     OPTIMIZERS["HybridModel"], patience=2, factor=0.5, verbose=True
        # # ),
        # "HybridModelV2s": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["HybridModelV2s"], patience=5, factor=0.5, verbose=True
        # ),
        # "HybridModelV2m": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["HybridModelV2m"], patience=5, factor=0.5, verbose=True
        # ),
        # "Xception": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["Xception"], patience=5, factor=0.5, verbose=True
        # ),
        # # "Inceptionv3": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["Inceptionv3"], patience=5, factor=0.5, verbose=True
        # ),
        # "DenseNet121": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["DenseNet121"], patience=5, factor=0.5, verbose=True
        # ),
        # "HybridInceptionV3": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["HybridInceptionV3"], patience=5, factor=0.5, verbose=True
        # ),
        # "HybridXception": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["HybridXception"], patience=5, factor=0.5, verbose=True
        # ),

        # "HybridModelv2b3": optim.lr_scheduler.ReduceLROnPlateau(
        #     OPTIMIZERS["HybridModelv2b3"], patience=5, factor=0.5, verbose=True
        # ),

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
        # "EfficientNetV2B3": EfficientNetV2B3,
        # "EfficientNetV2S": EfficientNetV2S,
        # "EfficientNetV2M": EfficientNetV2M,
        #"ViT": ViT,
        # "HybridModel": HybridModel,
        # "HybridModelV2s": HybridModelV2s,
        # "HybridModelV2m": HybridModelV2m,

        # "ViT": ViT,
        # "HybridModel": HybridModel,
        # "Xception": Xception,
        # "Inceptionv3": Inceptionv3,
        # "HybridXception": HybridXception,
        # "HybridInceptionv3": HybridInceptionV3,
        #"HybridModelv2b3": HybridModelv2b3,

        "MobileNetV3_large": MobileNetV3_large,
        "VGG16": VGG16,
        "ResNet50": ResNet50,
        "DenseNet121": DenseNet121,
        "MobileNetV3ViT": MobileNetV3ViT,
        "VGG16ViT": VGG16ViT,
        "ResNet50ViT": ResNet50ViT,
        "DenseNet121ViT": DenseNet121ViT,

    }

if NEW_DATASET:
    if AUGMENT:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_potatodata_Aug_True_015623.pth",
            # "ViT": "ViT_last_potatodata_Aug_True_134241_L2_dropout_hybrid.pth",
            # "HybridModel": "HybridModel_last_potatodata_Aug_True_204450_HT400k.pth",
            # "EfficientNetV2B3": "EfficientNetV2B3_potatodata_Aug_True_194849_EFF.pth",
            # "EfficientNetV2S": "EfficientNetV2S_potatodata_Aug_True_194849_EFF.pth",
            # "EfficientNetV2M": "EfficientNetV2M_potatodata_Aug_True_194849_EFF.pth",
            # "HybridModelV2s": "HybridModelV2s_potatodata_Aug_True_073227_EFF.pth",
            # "HybridModelV2m": "HybridModelV2m_potatodata_Aug_True_073227_EFF.pth",

            # "ViT": "ViT_potatodata_Aug_True_182227_ViT.pth",
            # "HybridModel": "HybridModel_potatodata_Aug_True_220918_ViT.pth",

            # "Xception": "Xception_potatodata_Aug_True_181149_CNNs.pth",
            # "Inceptionv3": "Inceptionv3_potatodata_Aug_True_162810_CNNs.pth",
            # "HybridXception": "HybridXception_potatodata_Aug_True_144722_CNNs.pth",
            # "HybridInceptionv3": "HybridInceptionV3_potatodata_Aug_True_185317_CNNs.pth",
            #"HybridModelv2b3": "HybridModelv2b3_potatodata_Aug_True_125450_CNNs.pth",

            "MobileNetV3_large": "MobileNetV3_large_potatodata_Aug_True_013313_CNNs.pth",
            "VGG16": "VGG16_potatodata_Aug_True_013313_CNNs.pth",
            "ResNet50": "ResNet50_potatodata_Aug_True_013313_CNNs.pth",
            "DenseNet121": "DenseNet121_potatodata_Aug_True_013313_CNNs.pth",
            "MobileNetV3ViT": "MobileNetV3ViT_potatodata_Aug_True_014437_CNNs.pth",
            "VGG16ViT": "VGG16ViT_potatodata_Aug_True_014437_CNNs.pth",
            "ResNet50ViT": "ResNet50ViT_potatodata_Aug_True_014437_CNNs.pth",
            "DenseNet121ViT": "DenseNet121ViT_potatodata_Aug_True_014437_CNNs.pth",

        }
    else:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_potatodata_Aug_False_082520.pth",
            # "ViT": "ViT_last_potatodata_Aug_False_134753_L2_dropout_hybrid.pth",
            # "HybridModel": "HybridModel_last_potatodata_Aug_False_205609_HT400k.pth",
            # "EfficientNetV2B3": "EfficientNetV2B3_potatodata_Aug_False_201907_EFF.pth",
            # "EfficientNetV2S": "EfficientNetV2S_potatodata_Aug_False_215124_EFF.pth",
            # "EfficientNetV2M": "EfficientNetV2M_potatodata_Aug_False_215124_EFF.pth",
            # "HybridModelV2s": "HybridModelV2s_potatodata_Aug_False_193355_EFF.pth",
            # "HybridModelV2m": "HybridModelV2m_potatodata_Aug_False_193355_EFF.pth",

            # "ViT": "ViT_potatodata_Aug_False_201457_ViT.pth",
            # "HybridModel": "HybridModel_potatodata_Aug_False_234841_ViT.pth",

            # "Xception": "Xception_potatodata_Aug_False_180720_CNNs.pth",
            # "Inceptionv3": "Inceptionv3_potatodata_Aug_False_225104_CNNs.pth",
            # "HybridXception": "HybridXception_potatodata_Aug_False_154325_CNNs.pth",
            # "HybridInceptionv3": "HybridInceptionV3_potatodata_Aug_False_122219_CNNs.pth",

            "MobileNetV3_large": "MobileNetV3_large_potatodata_Aug_False_013728_CNNs.pth",
            "VGG16": "VGG16_potatodata_Aug_False_013728_CNNs.pth",
            "ResNet50": "ResNet50_potatodata_Aug_False_013728_CNNs.pth",
            "DenseNet121": "DenseNet121_potatodata_Aug_False_013728_CNNs.pth",
            "MobileNetV3ViT": "MobileNetV3ViT_potatodata_Aug_False_015237_CNNs.pth",
            "VGG16ViT": "VGG16ViT_potatodata_Aug_False_015237_CNNs.pth",
            "ResNet50ViT": "ResNet50ViT_potatodata_Aug_False_015237_CNNs.pth",
            "DenseNet121ViT": "DenseNet121ViT_potatodata_Aug_False_015237_CNNs.pth",


        }
else:
    if AUGMENT:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_plantVillage_Aug_True_142228.pth",
            # "ViT": "ViT_last_plantVillage_Aug_True_223713_L2_dropout_hybrid.pth",
            # "HybridModel": "HybridModel_last_plantVillage_Aug_True_151547_HT400k.pth",

            # "EfficientNetV2B3": "EfficientNetV2B3_plantVillage_Aug_True_134512_EFF.pth",
            # "EfficientNetV2S": "EfficientNetV2S_plantVillage_Aug_True_134512_EFF.pth",
            # "EfficientNetV2M": "EfficientNetV2M_plantVillage_Aug_True_134512_EFF.pth",
            # "ViT": "ViT_plantVillage_Aug_True_033949_EFF.pth",
            # "HybridModelV2s": "HybridModelV2s_plantVillage_Aug_True_033949_EFF.pth",
            # "HybridModelV2m": "HybridModelV2m_plantVillage_Aug_True_033949_EFF.pth",

            "MobileNetV3_large": "MobileNetV3_large_plantVillage_Aug_True_103206_CNNs.pth",
            "VGG16": "VGG16_plantVillage_Aug_True_103206_CNNs.pth",
            "ResNet50": "ResNet50_plantVillage_Aug_True_103206_CNNs.pth",
            "DenseNet121": "DenseNet121ViT_plantVillage_Aug_True_103206_CNNs.pth",
            "MobileNetV3ViT": "MobileNetV3ViT_plantVillage_Aug_True_103206_CNNs.pth",
            "VGG16ViT": "VGG16ViT_plantVillage_Aug_True_103206_CNNs.pth",
            "ResNet50ViT": "ResNet50ViT_plantVillage_Aug_True_103206_CNNs.pth",
            "DenseNet121ViT": "DenseNet121ViT_plantVillage_Aug_False_103421_CNNs.pth",
        }
    else:
        SAVED_MODELS = {
            # "EfficientNetV2B3": "EfficientNetV2B3_last_plantVillage_Aug_False_141359.pth",
            # "ViT": "ViT_last_plantVillage_Aug_False_214638_L2_dropout_hybrid.pth",
            # "HybridModel": "HybridModel_last_plantVillage_Aug_False_082359_HT400k.pth",

            # "EfficientNetV2B3": "EfficientNetV2B3_plantVillage_Aug_False_125405_EFF.pth",
            # "EfficientNetV2S": "EfficientNetV2S_plantVillage_Aug_False_125405_EFF.pth",
            # "EfficientNetV2M": "EfficientNetV2M_plantVillage_Aug_False_125405_EFF.pth",
            # "ViT": "ViT_plantVillage_Aug_False_175130_EFF.pth",
            # "HybridModelV2s": "HybridModelV2s_plantVillage_Aug_False_175130_EFF.pth",
            # "HybridModelV2m": "HybridModelV2m_plantVillage_Aug_False_175130_EFF.pth",

            "MobileNetV3_large": "MobileNetV3_large_plantVillage_Aug_False_103421_CNNs.pth",
            "VGG16": "VGG16_plantVillage_Aug_False_103421_CNNs.pth",
            "ResNet50": "ResNet50_plantVillage_Aug_False_103421_CNNs.pth",
            "DenseNet121": "DenseNet121_plantVillage_Aug_False_103421_CNNs.pth",
            "MobileNetV3ViT": "MobileNetV3ViT_plantVillage_Aug_False_103421_CNNs.pth",
            "VGG16ViT": "VGG16ViT_plantVillage_Aug_False_103421_CNNs.pth",
            "ResNet50ViT": "ResNet50ViT_plantVillage_Aug_False_103421_CNNs.pth",
            "DenseNet121ViT": "DenseNet121ViT_plantVillage_Aug_False_103421_CNNs.pth",
        }

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    #name=f"plantv{time}_{DATATYPE}_train_Aug_{AUGMENT}",  # Train name # Added L2 regularization... 0.5
    name=f"Testpaper{time}_{DATATYPE}_test_Aug_{AUGMENT}",  # Test names
)
