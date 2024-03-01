import torch
import torch.nn as nn
import timm
from transformers import ViTModel
from torchinfo import summary


class HybridModel(nn.Module):
    def __init__(self, num_labels=7):  # Assuming FEATURES is defined as 7
        super(HybridModel, self).__init__()
        self.effnet = timm.create_model(
            "tf_efficientnetv2_b3", pretrained=True, features_only=True
        )

        # Unfreeze the last few layers of EfficientNet
        for param in list(self.effnet.parameters())[
            -20:
        ]:  # Unfreezing the last 20 parameters as an example
            param.requires_grad = True

        # identity layer
        self.effnet.identity = nn.Identity()

        self.fc = nn.Linear(1536, 3 * 224 * 224)  # Assuming ViT's embedding size is 768
        self.dropout_effnet = nn.Dropout(0.1)

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for param in self.vit.parameters():
            param.requires_grad = False  # Freeze ViT parameters to start

        # Optionally unfreeze some of the ViT layers
        for param in list(self.vit.encoder.layer[-1].parameters()):
            param.requires_grad = True

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        x = self.effnet(pixel_values)
        x = self.fc(x)
        x = x.view(x.shape[0], 3, 224, 224)  # Adapted for ViT
        # Feed the features into ViT
        outputs = self.vit(pixel_values=x)
        output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (logits, loss) if loss is not None else logits

