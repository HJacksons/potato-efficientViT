from torch import nn
from torchvision.models.vgg import VGG19_Weights
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.mobilenet import MobileNet_V2_Weights
from torchvision import models
from torchsummary import summary


# VGG19 model
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.classifier[6] = nn.Linear(4096, 7)

    def forward(self, x):
        return self.model(x)


# ResNet model
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.fc = nn.Linear(2048, 7)

    def forward(self, x):
        return self.model(x)


# MobileNetV2 model
class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.classifier[1] = nn.Linear(1280, 7)

    def forward(self, x):
        return self.model(x)
