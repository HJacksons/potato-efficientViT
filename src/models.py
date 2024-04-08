from torch import nn
from transformers import ViTModel
from utils import *
from timm import create_model
import torch
import torch.nn.functional as F
from torchsummary import summary
from torchinfo import summary


# EfficientNetV2B3 model used for reproducing the results
# class EfficientNetV2B3(nn.Module):
#     def __init__(self):
#         super(EfficientNetV2B3, self).__init__()
#         self.model = create_model("tf_efficientnetv2_b3.in21k", pretrained=True)
#         for param in self.model.parameters():
#             param.requires_grad = False
#         for name, param in self.model.named_parameters():
#             if "classifier" in name:
#                 param.requires_grad = True
#         self.model.classifier = nn.Linear(1536, FEATURES)
#
#     def forward(self, x):
#         # print("In EfficientNetV2B3 forward method")  # debug print
#
#         return self.model(x)

class EfficientNetV2B3(nn.Module):
    def __init__(self):
        super(EfficientNetV2B3, self).__init__()
        self.effnet = create_model("tf_efficientnetv2_b3.in21k", pretrained=True)

        for param in self.effnet.parameters():
            param.requires_grad = False

        self.effnet.classifier = nn.Sequential(  # Add a dropout layer before the final classifier layer
            nn.Dropout(0.5),
            nn.Linear(1536, FEATURES)
        )

    def forward(self, x):
        return self.effnet(x)


class EfficientNetV2S(nn.Module):
    def __init__(self):
        super(EfficientNetV2S, self).__init__()
        self.effnets = create_model("tf_efficientnetv2_s", pretrained=True)

        for param in self.effnets.parameters():
            param.requires_grad = False

        self.effnets.classifier = nn.Sequential(  # Add a dropout layer before the final classifier layer
            nn.Dropout(0.5),
            nn.Linear(1280, FEATURES)
        )

    def forward(self, x):
        return self.effnets(x)


class EfficientNetV2M(nn.Module):
    def __init__(self):
        super(EfficientNetV2M, self).__init__()
        self.effnetm = create_model("tf_efficientnetv2_m", pretrained=True)

        for param in self.effnetm.parameters():
            param.requires_grad = False

        self.effnetm.classifier = nn.Sequential(  # Add a dropout layer before the final classifier layer
            nn.Dropout(0.5),
            nn.Linear(1280, FEATURES)
        )

    def forward(self, x):
        return self.effnetm(x)


# ViT model
class ViT(nn.Module):
    def __init__(self, num_labels=FEATURES):
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for param in self.vit.parameters():
            param.requires_grad = False
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


class HybridModelv2b3(nn.Module):  # add more layers after combining the features
    def __init__(self, num_labels=FEATURES):
        super(HybridModelv2b3, self).__init__()

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
        self.fc1 = nn.Linear(1536 + 512, 1024)  # New FCL1
        self.fc2 = nn.Linear(1024, 512)  # New FCL2
        self.classifier = nn.Linear(512, num_labels)  # Adjusted to match the output of vit_linear

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
        # pass through new FCL1 and FCL2
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))

        output = self.classifier(combined)  # Get the final output

        return output


class HybridModelV2s(nn.Module):
    def __init__(self, num_labels=FEATURES):
        super(HybridModelV2s, self).__init__()

        # Part of EfficientNet
        self.effnet = create_model("tf_efficientnetv2_s", pretrained=True)
        self.effnet.classifier = nn.Identity()  # Remove the classifier
        for param in self.effnet.parameters():
            param.requires_grad = False  # Freeze the EfficientNet parameters

        # Part of ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False  # Freeze the ViT parameters

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1280 + 512, num_labels)  # Adjusted to match the output of vit_linear

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


class HybridModelV2m(nn.Module):
    def __init__(self, num_labels=FEATURES):
        super(HybridModelV2m, self).__init__()

        # Part of EfficientNet
        self.effnet = create_model("tf_efficientnetv2_m", pretrained=True)
        self.effnet.classifier = nn.Identity()  # Remove the classifier
        for param in self.effnet.parameters():
            param.requires_grad = False  # Freeze the EfficientNet parameters

        # Part of ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False  # Freeze the ViT parameters

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1280 + 512, num_labels)  # Adjusted to match the output of vit_linear

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


# https://arxiv.org/abs/1610.02357
class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        self.model = create_model("xception", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, FEATURES)
        )

    def forward(self, x):
        return self.model(x)


# https://arxiv.org/abs/1512.00567
class Inceptionv3(nn.Module):
    def __init__(self):
        super(Inceptionv3, self).__init__()
        self.model = create_model("inception_v3", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, FEATURES)
        )

    def forward(self, x):
        return self.model(x)


# https://arxiv.org/abs/1608.06993
class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = create_model("densenet121", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, FEATURES)
        )

    def forward(self, x):
        return self.model(x)


class DenseNet121ViT(nn.Module):
    def __init__(self):
        super(DenseNet121ViT, self).__init__()
        self.densenet = create_model("densenet121", pretrained=True)
        self.densenet.classifier = nn.Identity()
        for param in self.densenet.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024 + 512, FEATURES)

    def forward(self, x):
        # Extract features using DenseNet121
        densenet_output = self.densenet.forward_features(x)
        densenet_output = torch.flatten(densenet_output, start_dim=2)
        densenet_output = densenet_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)

        # Combine the outputs
        combined = torch.cat((densenet_output, vit_output), dim=1)

        # Apply dropout
        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output

class MobileNetV3_large(nn.Module):
    def __init__(self):
        super(MobileNetV3_large, self).__init__()
        self.model = create_model("mobilenetv3_large_100", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, FEATURES)
        )

    def forward(self, x):
        return self.model(x)

class MobileNetV3ViT(nn.Module):
    def __init__(self):
        super(MobileNetV3ViT, self).__init__()
        self.mobilenetv3 = create_model("mobilenetv3_large_100", pretrained=True)
        self.mobilenetv3.classifier = nn.Identity()
        for param in self.mobilenetv3.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(960 + 512, FEATURES)

    def forward(self, x):
        # Extract features using MobileNetV3
        mobilenetv3_output = self.mobilenetv3.forward_features(x)
        mobilenetv3_output = torch.flatten(mobilenetv3_output, start_dim=2)
        mobilenetv3_output = mobilenetv3_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)

        # Combine the outputs
        combined = torch.cat((mobilenetv3_output, vit_output), dim=1)

        # Apply dropout
        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = create_model("vgg16", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, FEATURES)
        )

    def forward(self, x):
        return self.model(x)

class VGG16ViT(nn.Module):
    def __init__(self):
        super(VGG16ViT, self).__init__()
        self.vgg16 = create_model("vgg16", pretrained=True)
        self.vgg16.classifier = nn.Identity()
        for param in self.vgg16.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(512 + 512, FEATURES)

    def forward(self, x):
        # Extract features using VGG16
        vgg16_output = self.vgg16.forward_features(x)
        vgg16_output = torch.flatten(vgg16_output, start_dim=2)
        vgg16_output = vgg16_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)
        print("mobilenetv3_output shape:", vgg16_output.shape)
        print("vit_output shape:", vit_output.shape)

        # Combine the outputs
        combined = torch.cat((vgg16_output, vit_output), dim=1)

        # Apply dropout
        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = create_model("resnet50.a1_in1k", pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, FEATURES)
        )

    def forward(self, x):
        return self.model(x)

class ResNet50ViT(nn.Module):
    def __init__(self):
        super(ResNet50ViT, self).__init__()
        self.resnet50 = create_model("resnet50.a1_in1k", pretrained=True)
        self.resnet50.fc = nn.Identity()
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2048 + 512, FEATURES)

    def forward(self, x):
        # Extract features using ResNet50
        resnet50_output = self.resnet50.forward_features(x)
        resnet50_output = torch.flatten(resnet50_output, start_dim=2)
        resnet50_output = resnet50_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)

        # Combine the outputs
        combined = torch.cat((resnet50_output, vit_output), dim=1)

        # Apply dropout
        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output

class HybridInceptionV3(nn.Module):
    def __init__(self):
        super(HybridInceptionV3, self).__init__()
        self.inception = create_model("inception_v3", pretrained=True)
        self.inception.fc = nn.Identity()
        for param in self.inception.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2048 + 512, FEATURES)

    def forward(self, x):
        # Extract features using InceptionV3
        inception_output = self.inception.forward_features(x)
        inception_output = torch.flatten(inception_output, start_dim=2)
        inception_output = inception_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)

        # Combine the outputs
        combined = torch.cat((inception_output, vit_output), dim=1)

        # Apply dropout
        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output


class HybridXception(nn.Module):
    def __init__(self):
        super(HybridXception, self).__init__()
        self.xception = create_model("xception", pretrained=True)
        self.xception.fc = nn.Identity()
        for param in self.xception.parameters():
            param.requires_grad = False
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_linear = nn.Linear(self.vit.config.hidden_size, 512)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2048 + 512, FEATURES)

    def forward(self, x):
        # Extract features using Xception
        xception_output = self.xception.forward_features(x)
        xception_output = torch.flatten(xception_output, start_dim=2)
        xception_output = xception_output.mean(dim=2)

        # Feed the input into ViT
        vit_output = self.vit(pixel_values=x)['last_hidden_state'][:, 0]
        vit_output = self.vit_linear(vit_output)

        # Combine the outputs
        combined = torch.cat((xception_output, vit_output), dim=1)

        # Apply dropout
        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output


#
# mod = ResNet50ViT()
# summary(mod, input_size=(1, 3, 224, 224))
