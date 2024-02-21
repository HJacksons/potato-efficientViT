from torch import nn
from torchvision.models.vgg import VGG19_Weights
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.mobilenet import MobileNet_V2_Weights
from transformers import ViTModel, ViTForImageClassification
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


# ViT model
class ViT(nn.Module):
    def __init__(self, num_labels=7):
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None
