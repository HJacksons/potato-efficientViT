from torch import nn
from transformers import ViTModel
from utils import *
import timm
from torchsummary import summary
from torchinfo import summary


# EfficientNetV2B3 model
class EfficientNetV2B3(nn.Module):
    def __init__(self):
        super(EfficientNetV2B3, self).__init__()
        self.model = timm.create_model("tf_efficientnetv2_b3.in21k", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
        self.model.classifier = nn.Linear(1536, FEATURES)

    def forward(self, x):
        # print("In EfficientNetV2B3 forward method")  # debug print

        return self.model(x)


# ViT model
class ViT(nn.Module):
    def __init__(self, num_labels=FEATURES):
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for param in self.vit.parameters():
            param.requires_grad = False
        for name, param in self.vit.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
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


# The hybrid model
class HybridModel(nn.Module):
    def __init__(self, num_labels=FEATURES):
        super(HybridModel, self).__init__()
        self.effnet = timm.create_model("tf_efficientnetv2_b3.in21k", pretrained=True)
        # Freeze the EfficientNetV2B3 model
        for param in self.effnet.parameters():
            param.requires_grad = False
        # Replace the classifier with a no op (identity) to get the features
        self.effnet.classifier = nn.Identity()

        # Add a fully-connected layer to transform the output shape
        self.fc = nn.Linear(
            1536, 3 * 224 * 224
        )  # Replace 3 with the number of channels expected by ViT
        self.dropout_effnet = nn.Dropout(0.5)

        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=num_labels
        )
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.encoder.layer[-1].parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        # Extract features from EfficientNetV2B3
        x = self.effnet(pixel_values)
        # Transform the output shape
        x = self.fc(x)
        x = x.view(x.shape[0], 3, 224, 224)  # Reshape to match ViT's input shape
        # Feed the features into ViT
        outputs = self.vit(pixel_values=x)
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
