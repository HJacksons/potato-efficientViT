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
from PIL import Image


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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None
        self.model.eval()

        # Ensure hooks are correctly attached
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.feature_maps = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def get_feature_maps_hook(self, module, input, output):
        self.feature_maps = output

    def generate_heatmap(self, input_image, class_idx):
        # Forward
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0]

        # Zero grads
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(device)
        one_hot_output[0][class_idx] = 1
        output.backward(gradient=one_hot_output)

        # Make sure to move the gradients to CPU before converting to NumPy
        gradient = self.gradients.data.cpu().numpy()[0]  # Corrected line
        weight = np.mean(gradient, axis=(1, 2))
        feature_map = self.feature_maps.data.cpu().numpy()[0]  # Also ensure feature maps are on CPU

        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        for i, w in enumerate(weight):
            cam += w * feature_map[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def apply_colormap_on_image(org_img, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(org_img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# Visualization of Grad-CAM
fig, axs = plt.subplots(8, 2, figsize=(10, 40))
classes = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']  # Update this with your actual class names

for batch_idx, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Get the model output
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    original_images = []
    cam_images = []

    for i in range(min(inputs.size(0), 8)):  # Ensures i < 8 for visualization purposes
        img = inputs.data[i].cpu().numpy().transpose((1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        grad_cam = GradCAM(model, model.model.features[-1])  # Ensure this matches your model's structure
        heatmap = grad_cam.generate_heatmap(inputs[i].unsqueeze(0), preds[i].item())  # Use .item() for class index

        cam_img = apply_colormap_on_image(img, heatmap)

        # Construct caption with true and predicted classes
        true_class = classes[labels[i].item()]  # Assuming 'classes' is a list of class names
        pred_class = classes[preds[i].item()]
        pred_score = outputs[i][preds[i]]    #
        caption = f"True: {true_class} | Pred: {pred_class} | PredScore: {pred_score :.4f}"

        # Add images with captions to lists
        original_images.append(wandb.Image(img, caption=caption))
        cam_images.append(wandb.Image(cam_img, caption=caption))

    # Log all images as a group
    wandb.log({"Original Images": original_images, "Grad-CAM Overlays": cam_images})

    break  # Break after the first batch

plt.tight_layout()
plt.show()