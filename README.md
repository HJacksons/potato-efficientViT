# Deep Learning for Potato Plant Disease Detection and Identification

This project implements and evaluates deep learning models for potato plant disease detection and identification using image datasets. It provides a framework for training and testing various Convolutional Neural Network (CNN) and Vision Transformer (ViT) based architectures, including hybrid models.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Models](#models)
- [Logging](#logging)

## Project Overview

This project focuses on the development and evaluation of a novel hybrid deep learning architecture, **EfficientNetV2B3ViT**, for advanced potato plant disease detection and identification from images. The work presented here forms the basis of a research paper, showcasing the model's superior performance and capabilities. While the primary focus is on `EfficientNetV2B3ViT`, the framework also supports experimentation with other Convolutional Neural Network (CNN) and Vision Transformer (ViT) based architectures.

### Publication Details

This research has been published in:
- **Journal:** BMC Plant Biology
- **DOI:** [10.1186/s12870-025-06679-4](https://doi.org/10.1186/s12870-025-06679-4)
- **Title:** Potato plant disease detection: leveraging hybrid deep learning models.

## Features

- Support for multiple datasets (e.g., `potatodata`, `plantVillage`).
- Implementation of various CNN and ViT models, including hybrid architectures like `EffNetViT`.
- Configurable training parameters (epochs, learning rate, batch size, etc.).
- Data augmentation options to improve model generalization.
- Training and testing scripts.
- Logging of experiments using Weights & Biases (wandb).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Ensure Poetry is installed:**
    If you don't have Poetry installed, follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

3.  **Install dependencies using Poetry:**
    Navigate to the project root (where `pyproject.toml` is located) and run:
    ```bash
    poetry install
    ```
    This will create a virtual environment if one doesn't exist and install all the dependencies specified in `pyproject.toml`.

4.  **Activate the virtual environment (if needed):**
    Poetry often manages virtual environments automatically. To run commands within the project's environment, you can use `poetry run <your_command>` (e.g., `poetry run python src/train.py`).
    Alternatively, you can activate the environment directly using `poetry shell`.

5.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your Weights & Biases API key and project details:
    ```
    WANDB_KEY="your_wandb_api_key"
    WANDB_PROJECT="your_wandb_project_name"
    WANDB_ENTITY="your_wandb_entity"
    ```

## Dataset

The project is configured to work with datasets placed in the `data/` directory. The actual data files for `potatodata` and `plantVillage` are not version-controlled with Git (as indicated by the project's `.gitignore` file) and need to be obtained separately.

**Setup:**
1.  Create the directories `data/potatodata` and `data/plantVillage` if they don't exist.
2.  Populate these directories with your respective image datasets.
3.  Ensure each dataset is structured with subdirectories for each class.

The specific dataset to be used during runtime is set in `src/configurations.py` via the `DATA` variable (e.g., `DATA = "../data/potatodata"`).

## Configuration

All major settings for training, testing, model selection, and dataset paths are managed in `src/configurations.py`. Key configurations include:

-   `DATA`: Path to the dataset.
-   `TEST_SIZE`, `VALI_SIZE`: Ratios for splitting the dataset.
-   `BATCH_SIZE`: Batch size for training and evaluation.
-   `CLASSES`: Automatically determined from the dataset directory.
-   `DEVICE`: Automatically detects CUDA availability.
-   `CRITERION`: Loss function (e.g., `nn.CrossEntropyLoss()`).
-   `EPOCHS`: Number of training epochs.
-   `lr`: Learning rate.
-   `TRAINING`: Boolean flag to switch between training (`True`) and testing (`False`) modes.
-   `AUGMENT`: Boolean flag to enable/disable data augmentation.
-   `DATATYPE`: Name of the dataset being used (e.g., "potatodata").
-   `NEW_DATASET`: Flag for specific saved model paths, often used to differentiate between experiments.
-   `MODELS`: Dictionary defining the models to be used.
-   `OPTIMIZERS`: Dictionary defining optimizers for each model.
-   `SCHEDULER`: Dictionary defining learning rate schedulers.
-   `SAVED_MODELS`: Dictionary mapping model names to their saved checkpoint file paths, conditional on `NEW_DATASET` and `AUGMENT` flags.

Modify this file to experiment with different settings.

## Usage

Ensure you have configured `src/configurations.py` appropriately before running the scripts.

### Training

To train a model:
1.  Set `TRAINING = True` in `src/configurations.py`.
2.  Configure other relevant parameters like `EPOCHS`, `lr`, `MODEL` to train, `DATATYPE`, etc.
3.  Run the training script:
    ```bash
    python src/train.py
    ```
    Trained models will be saved, and progress will be logged to Weights & Biases.

### Testing

To test a trained model:
1.  Set `TRAINING = False` in `src/configurations.py`.
2.  Ensure the `SAVED_MODELS` dictionary in `src/configurations.py` points to the correct trained model checkpoint for the `EfficientNetV2B3ViT` (often keyed as `EffNetViT`) or other model you wish to test.
3.  Specify the model to test in the `MODELS` dictionary (it should reference the class, not an instance).
4.  Run the testing script:
    ```bash
    python src/test.py
    ```
    Evaluation metrics will be printed, and results can also be logged to Weights & Biases.

## Models

The foundation of this project is the **`EfficientNetV2B3ViT`** hybrid model which combines the strengths of EfficientNetV2B3 and a Vision Transformer. This model, defined in `src/models.py` (and typically referenced as `EffNetViT` in `src/configurations.py`), is the primary subject of the research publication due to its promising results in potato plant disease detection and identification.

While `EfficientNetV2B3ViT` is the main focus, the framework also includes implementations of other models for comparative analysis, such as:
- Standard Vision Transformer (`ViT`)
- `EfficientNetV2B3`
- `MobileNetV3_large`
- `VGG16`
- `ResNet50`
- `DenseNet121`
- And their respective ViT-hybrid counterparts (e.g., `MobileNetV3ViT`, `VGG16ViT`).

These can be configured and experimented with via `src/configurations.py`.

## Logging

Experiment tracking and results are logged using [Weights & Biases](https://wandb.ai). Ensure your `.env` file is configured with your W&B credentials. The run name for logging is dynamically generated in `src/configurations.py` based on the timestamp, dataset type, and augmentation status.

---

*This README provides a general guide. Modify paths and commands as per your specific setup.*