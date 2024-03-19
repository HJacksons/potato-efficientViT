from torch import nn
from transformers import ViTModel
from utils import *
from timm import create_model
import torch
import torch.nn.functional as F

# from torchsummary import summary
from torchinfo import summary


# EfficientNetV2B3 model
class EfficientNetV2B3(nn.Module):
    def __init__(self):
        super(EfficientNetV2B3, self).__init__()
        self.model = create_model("tf_efficientnetv2_b3.in21k", pretrained=True)
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


# # The hybrid model
# class HybridModel(nn.Module):
#     def __init__(self, num_labels=FEATURES):
#         super(HybridModel, self).__init__()
#         self.effnet = timm.create_model("tf_efficientnetv2_b3.in21k", pretrained=True)
#         # Freeze the EfficientNetV2B3 model
#         for param in self.effnet.parameters():
#             param.requires_grad = False
#         # unfreeze the last layer few layers of the model, may be 3
#         for i, param in enumerate(self.effnet.parameters()):
#             if i >= len(list(self.effnet.parameters())) - 3:
#                 param.requires_grad = True
#
#         # Replace the classifier with a no op (identity) to get the features
#         self.effnet.classifier = nn.Identity()
#
#         # Add a fully-connected layer to transform the output shape
#         self.fc = nn.Linear(
#             1536, 3 * 224 * 224
#         )  # Replace 3 with the number of channels expected by ViT
#         self.dropout_effnet = nn.Dropout(
#             0.2
#         )  # added dropout later to avoid overfitting
#
#         self.vit = ViTModel.from_pretrained(
#             "google/vit-base-patch16-224-in21k", num_labels=num_labels
#         )
#         for param in self.vit.parameters():
#             param.requires_grad = False
#         for param in self.vit.encoder.layer[-1].parameters():
#             param.requires_grad = True
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
#
#     def forward(self, pixel_values, labels=None):
#         # Extract features from EfficientNetV2B3
#         x = self.effnet(pixel_values)
#         # Transform the output shape
#         x = self.fc(x)
#         x = x.view(x.shape[0], 3, 224, 224)  # Reshape to match ViT's input shape
#         # Feed the features into ViT
#         outputs = self.vit(pixel_values=x)
#         output = self.dropout(outputs.last_hidden_state[:, 0])
#         logits = self.classifier(output)
#
#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#         if loss is not None:
#             return logits, loss.item()
#         else:
#             return logits, None


# class HybridModel(nn.Module): # currently training
#     def __init__(self, num_labels=FEATURES):
#         super(HybridModel, self).__init__()
#
#         # Part of EfficientNet
#         self.effnet = create_model("tf_efficientnetv2_b3.in21k", pretrained=True)
#         self.effnet.classifier = nn.Identity()  # Remove the original classifier
#
#         # Freeze all EfficientNet layers except the last block
#         for param in self.effnet.parameters():
#             param.requires_grad = False
#         for param in self.effnet.blocks[-1].parameters():
#             param.requires_grad = True
#
#
#         # Part of ViT
#         self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#         self.effnet_linear = nn.Linear(1536, 1024)
#         self.vit_linear = nn.Linear(self.vit.config.hidden_size, 1024)
#
#         # Freeze ViT layers for now
#         for param in self.vit.parameters():
#             param.requires_grad = False
#         for param in self.vit.encoder.layer[-1].parameters():
#             param.requires_grad = True
#
#         # Feature fusion layer (replaces FC1)
#         self.fusion = nn.Sequential(
#             nn.Linear(2048, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2)
#         )
#
#         # Additional layer (replaces FC2)
#         self.fc = nn.Linear(1024, 1024)
#
#         # Final classifier
#         self.classifier = nn.Linear(1024, num_labels)
#
#     def forward(self, x):
#         # Extract features using EfficientNet (up to last block)
#         effnet_output = self.effnet.forward_features(x)
#         effnet_output = torch.flatten(effnet_output, start_dim=2)  # Flatten the output
#         effnet_output = effnet_output.mean(dim=2)  # Global average pooling
#         effnet_output = self.effnet_linear(effnet_output)  # Add this line
#
#         # Feed the input into ViT
#         vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
#         vit_output = self.vit_linear(vit_output)  # Reduce dimensionality
#
#         # Combine features (earlier fusion)
#         combined = torch.cat((effnet_output, vit_output), dim=1)
#         combined = self.fusion(combined)  # Feature fusion layer
#
#         # Pass through additional layer
#         x = F.relu(self.fc(combined))
#
#         # Get the final output
#         output = self.classifier(x)
#
#         return output


class HybridModel(nn.Module):
    def __init__(self, num_labels=FEATURES):
        super(HybridModel, self).__init__()

        # Part of EfficientNet
        self.effnet = create_model("tf_efficientnetv2_b3.in21k", pretrained=True)
        self.effnet.classifier = nn.Identity()  # Remove the classifier
        for param in self.effnet.parameters():
            param.requires_grad = False  # Freeze the EfficientNet parameters

        # Part of ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False  # Freeze the ViT parameters

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1536 + 512, num_labels)  # Adjusted to match the output of vit_linear

    def forward(self, x):
        # Extract features using EfficientNet
        effnet_output = self.effnet.forward_features(x)
        effnet_output = torch.flatten(effnet_output, start_dim=2)  # Flatten the output
        effnet_output = effnet_output.mean(dim=2)  # Global average pooling

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)  # Reduce the dimensionality of the ViT output

        # Combine the outputs
        combined = torch.cat((effnet_output, vit_output), dim=1)

        # Apply dropout
        combined = self.dropout(combined)

        output = self.classifier(combined)  # Get the final output

        return output

# mod = HybridModel()
# summary(mod, input_size=(1, 3, 224, 224))