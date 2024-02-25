from torchvision import transforms
import os


@staticmethod
def get_transforms_for_model(augment):
    if augment:
        data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
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
DATA = "../data/potatodata"
CLASSES = sorted(os.listdir(DATA))
FEATURES = len(CLASSES)
