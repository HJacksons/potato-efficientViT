import timm
import torch.nn as nn
from torchsummary import summary


# Load the pre-trained model
efficientnet = timm.create_model("tf_efficientnetv2_b3", pretrained=True)

# Freeze all layers
for param in efficientnet.parameters():
    param.requires_grad = False

# Unfreeze the Fully Connected Layers (Classifier)
# Assuming the classifier layer's parameters contain 'classifier' in their names
for name, param in efficientnet.named_parameters():
    if "classifier" in name:
        param.requires_grad = True

model = efficientnet
# Replace the classifier layer with a new one
model.classifier = nn.Linear(1536, 7)

# Print the model
print(summary(model, (3, 224, 224)))

# Load the pre-trained model
vit = timm.create_model("vit_base_patch16_224", pretrained=True)

# Freeze all layers
for param in vit.parameters():
    param.requires_grad = False

# Unfreeze the head of the model
# Assuming the head's parameters contain 'head' in their names
for name, param in vit.named_parameters():
    if "head" in name:
        param.requires_grad = True

# Replace the head with a new one
vit.head = nn.Linear(768, 7)

# Print the model
# print(summary(vit, (3, 224, 224)))
