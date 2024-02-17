from vgg import VGG
import torch
from torch import nn
from dataset import Dataset
import logging
import os
import wandb
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import cv2



# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up wandb
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

# We are going to train the vgg on our dataset

# Load the dataset
dataset = Dataset()
train_loader, vali_loader, test_loader = dataset.prepare_dataset()
model = VGG()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the model
model.load_state_dict(torch.load(f"vgg_model_50.pth"))
model.eval()

# implement Grad-CAM and visualize the results

# Get the image and label
img, label = next(iter(test_loader))
img, label = img.to(device), label.to(device)

# Get the class index
class_idx = label[0].item()

# Get the model's prediction
pred = model(img)
pred = pred.argmax(dim=1)

# Get the gradient of the output with respect to the parameters of the model
pred[:, class_idx].backward()

# Pull the gradients out of the model
gradients = model.get_activations_gradient()

# Pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# Get the activations of the last convolutional layer
activations = model.get_activations(img).detach()

# Weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]

# Average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# ReLU on top of the heatmap
heatmap = np.maximum(heatmap.cpu(), 0)

# Normalize the heatmap
heatmap /= torch.max(heatmap)

# Draw the heatmap
plt.matshow(heatmap.squeeze())
plt.show()

# Load the original image
original_image = img[0].cpu().numpy().transpose((1, 2, 0))
original_image = np.clip(original_image, 0, 1)

# Resize the heatmap to be the same size as the original image
heatmap = cv2.resize(heatmap.numpy(), (original_image.shape[1], original_image.shape[0]))

# Convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# Apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Superimpose the heatmap on the original image
superimposed_img = heatmap * 0.4 + original_image * 255
superimposed_img /= np.max(superimposed_img)

# Display the superimposed image
plt.imshow(superimposed_img)
plt.show()
# Save the superimposed image
cv2.imwrite("superimposed_img.jpg", superimposed_img)
wandb.log({"Grad-CAM": [wandb.Image("superimposed_img.jpg", caption="Grad-CAM")]})
