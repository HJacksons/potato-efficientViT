import torch
from vit_rollout import VITAttentionRollout
from transformers import ViTModel
from torch import nn
from utils import FEATURES
from dataset import Dataset
from torchvision.utils import save_image


class ViT(nn.Module):
    def __init__(self, num_labels=FEATURES):
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        for param in self.vit.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

        for layer in self.vit.encoder.layer:
            layer.attention.output.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def forward(self, pixel_values, labels=None):
        print(pixel_values.shape, type(pixel_values))  # Add this line

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

    def get_attention_map(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            _ = self(input_tensor)

        # Use the VITAttentionRollout class from the previous code
        rollout = VITAttentionRollout(self, discard_ratio=0.9)
        self.train()
        attention_map = rollout(input_tensor)
        self.eval()

        return attention_map


model = ViT()
model.load_state_dict(torch.load("ViT_last_potatodata_Aug_False_134753_L2_dropout_hybrid.pth", map_location=torch.device('cpu')))
model.eval()

# Load your test data
data = Dataset()
_, _, test_loader = data.prepare_dataset()

# Set the number of images to explain
num_images = 4

# Iterate over the test data
for i, (input_tensors, labels) in enumerate(test_loader):
    # Generate an attention map for each input tensor
    for j in range(input_tensors.size(0)):  # Iterate over the batch dimension
        input_tensor = input_tensors[j]  # Select one image from the batch
        attention_map = model.get_attention_map(input_tensor.unsqueeze(0))  # Add the batch dimension back

        # Now you can do something with the attention map, like saving it to a file
        attention_map_tensor = torch.from_numpy(attention_map)[None, :, :]
        save_image(attention_map_tensor, f"attention_map_{i}_{j}.png")

        # Break the loop after num_images images
        if (i * len(input_tensors) + j + 1) >= num_images:
            break
    if (i * len(input_tensors) + j + 1) >= num_images:
        break