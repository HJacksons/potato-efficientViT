from torch import nn
from torchvision.models.vgg import VGG19_Weights
from torchvision import models
from torchsummary import summary


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.classifier[6] = nn.Linear(4096, 7)

    def forward(self, x):
        return self.model(x)
