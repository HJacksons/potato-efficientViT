import os
import random
import cv2
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset
import logging
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import gc
import numpy as np



logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)

DATA = "../data/potatodata"
TEST_SIZE = 0.3
VALI_SIZE = 0.5
RANDOM_STATE = 42
BATCH_SIZE = 32
CLASSES = os.listdir(DATA)
# print classes with their index
# for i, class_name in enumerate(CLASSES):
#     print(f"{i}: {class_name}")


class Dataset:
    def __init__(
            self,
            dataset=DATA,
            test_size=TEST_SIZE,
            vali_size=VALI_SIZE,
            random_state=RANDOM_STATE,

    ):
        self.dataset_name = dataset
        self.test_size = test_size
        self.vali_size = vali_size
        self.random_state = random_state
        self.data_transforms = self.get_transforms_for_model()

    @staticmethod
    def get_transforms_for_model():
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(hue=0.021, saturation=0.8, brightness=0.43),
                transforms.RandomAffine(
                    degrees=0, translate=(0.13, 0.13), scale=(0.95, 1.05)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return data_transforms

    def prepare_dataset(self):
        # Load dataset
        dataset = datasets.ImageFolder(
            self.dataset_name, transform=self.data_transforms
        )

        # Get targets/labels from the dataset
        targets = np.array([s[1] for s in dataset.samples])

        # Split the dataset into train, validation, and test sets
        train_indices, temp_indices = train_test_split(
            np.arange(len(targets)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=targets,
        )
        vali_indices, test_indices = train_test_split(
            temp_indices,
            test_size=self.vali_size,
            random_state=self.random_state,
            stratify=targets[temp_indices],
        )

        # Create subsets from the indices
        train_dataset = Subset(dataset, train_indices)
        vali_dataset = Subset(dataset, vali_indices)
        test_dataset = Subset(dataset, test_indices)

        # Print the number of samples in each set
        # logging.info(f"Number of samples in the training set: {len(train_dataset)}")
        # logging.info(f"Number of samples in the validation set: {len(vali_dataset)}")
        # logging.info(f"Number of samples in the test set: {len(test_dataset)}")

        # Create data loaders
        train_dl = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        vali_dl = DataLoader(
            vali_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_dl = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # visualize 3 samples from each of the train, validation, and test sets with label by name of the class
        def denormalize(img):
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            im = img * std[:, None, None] + mean[:, None, None]  # Apply to each channel
            im = np.clip(im, 0, 1)  # Ensure the values are between 0 and 1
            return im

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i, loader in enumerate([train_dl, vali_dl, test_dl]):
            for j in range(3):
                idx = random.randint(0, len(loader.dataset) - 1)
                image, label = loader.dataset[idx]
                # Convert tensor image to numpy and denormalize
                image = image.numpy()  # Convert from torch tensor to numpy array
                image = denormalize(image)  # Denormalize the image
                image = np.transpose(image, (1, 2, 0))  # Change from (C, H, W) to (H, W, C) for plotting

                axes[i, j].imshow(image)
                axes[i, j].set_title(f"{CLASSES[label]}")
                axes[i, j].axis("off")
        plt.show()

        return train_dl, vali_dl, test_dl


prepare = Dataset()
train_loader, vali_loader, test_loader = prepare.prepare_dataset()
