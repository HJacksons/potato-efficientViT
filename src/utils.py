from torchvision import transforms
import os


@staticmethod
def get_transforms_for_model(augment):
    if augment:
        data_transforms = transforms.Compose( # trying to reproduce the same results
            [
                transforms.Resize(
                    (224, 224)
                ),  # Resize images to 224x224
                transforms.RandomHorizontalFlip(),  # Horizontal flipping
                transforms.RandomVerticalFlip(),  # Vertical flipping
                transforms.RandomRotation(15),  # Rotation
                transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),  # Zooming
                transforms.ColorJitter(brightness=0.2),  # Brightness adjustment
                transforms.RandomAffine(
                    0, translate=(0.1, 0.1)
                ),  # Horizontal and vertical shifting
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # Normalization
            ]
        )
        # data_transforms = transforms.Compose(
        #     [
        #         transforms.Resize(256),  # Resize the shortest side of the image to 256
        #         transforms.RandomResizedCrop(
        #             224,
        #             scale=(0.5, 1.5),  # Adjust scale for more intense zooming
        #             ratio=(0.75, 1.33),
        #             interpolation=transforms.InterpolationMode.BICUBIC,
        #         ),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.RandomRotation(30),  # Increase rotation range
        #         transforms.ColorJitter(
        #             brightness=0.3, contrast=0.2, saturation=0.2, hue=0.02
        #         ),
        #         # Add contrast and saturation adjustment
        #         transforms.RandomAffine(
        #             degrees=0,
        #             translate=(0.2, 0.2),  # Increase translate values for more shifting
        #             scale=(
        #                 0.8,
        #                 1.2,
        #             ),  # Adjust scale values for additional zooming effect
        #         ),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )
    else:
        data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return data_transforms


# Augment True, False
DATA = "../data/plantVillage"
CLASSES = sorted(os.listdir(DATA))
FEATURES = len(CLASSES)
