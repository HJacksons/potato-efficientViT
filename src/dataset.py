from configurations import *
from utils import get_transforms_for_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import logging
from torchvision import datasets
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


class Dataset:
    def __init__(
        self,
        dataset=DATA,
        test_size=TEST_SIZE,
        vali_size=VALI_SIZE,
        random_state=RANDOM_STATE,
        augment=AUGMENT,
    ):
        self.dataset_name = dataset
        self.test_size = test_size
        self.vali_size = vali_size
        self.random_state = random_state
        self.augment = augment
        self.train_transforms = get_transforms_for_model(augment)
        self.other_transforms = get_transforms_for_model(False)
        print(augment)

    def prepare_dataset(self):
        # Load dataset
        dataset = datasets.ImageFolder(self.dataset_name)

        # Get targets/labels from the dataset
        targets = np.array([s[1] for s in dataset.samples])

        # Split the dataset into train and test sets
        train_indices, test_indices = train_test_split(
            np.arange(len(targets)),
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_indices, vali_indices = train_test_split(
            train_indices,
            test_size=self.vali_size,
            random_state=self.random_state,
        )

        # Create subsets from the indices
        train_dataset = Subset(dataset, train_indices)
        vali_dataset = Subset(dataset, vali_indices)
        test_dataset = Subset(dataset, test_indices)
        # print("Train, valid, test", len(train_dataset), len(vali_dataset), len(test_dataset))
        if self.augment:
            train_dataset.dataset.transform = self.train_transforms
        vali_dataset.dataset.transform = self.other_transforms
        test_dataset.dataset.transform = self.other_transforms

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

        return train_dl, vali_dl, test_dl
