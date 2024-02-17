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

# # Model
# model = VGG()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
#
# # Train the model, log train loss, train accuracy, validation loss, validation accuracy
epochs = 50
#
# for epoch in range(epochs):
#     model.train()
#     total_loss, total_correct, total_samples = 0, 0, 0
#
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total_correct += (predicted == labels).sum().item()
#         total_samples += labels.size(0)
#
#     avg_loss = total_loss / len(train_loader)
#     avg_accuracy = total_correct / total_samples
#     logging.info(
#         f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}"
#     )
#     wandb.log({"Train Loss": avg_loss, "Train Accuracy": avg_accuracy})
#
#     model.eval()
#     total_loss, total_correct, total_samples = 0, 0, 0
#
#     with torch.no_grad():
#         for images, labels in vali_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total_correct += (predicted == labels).sum().item()
#             total_samples += labels.size(0)
#
#     avg_loss = total_loss / len(vali_loader)
#     avg_accuracy = total_correct / total_samples
#     logging.info(
#         f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}"
#     )
#     wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": avg_accuracy})
#
# # Save the model
# torch.save(model.state_dict(), f"vgg_model_{epochs}.pth")
# wandb.save(f"vgg_model_{epochs}.pth")
# logging.info("Model saved.")

##########################

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.model.eval()

    def get_gradients(self, grad):
        self.gradients = grad

    def forward_pass_with_hooks(self, input_image):
        conv_output = None  # The output of the target layer

        # Register hook to capture the gradients of the target layer
        hook_backward = self.target_layer.register_backward_hook(self.get_gradients)

        # Forward pass
        for name, module in self.model.named_children():
            input_image = module(input_image)
            if name == self.target_layer:
                # Register hook to capture the output of the target layer
                hook_forward = module.register_forward_hook(lambda module, input, output: setattr(self, "conv_output", output))
                break

        return conv_output, input_image

    def generate_heatmap(self, input_image, class_idx):
        # Forward
        conv_output, model_output = self.forward_pass_with_hooks(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(input_image.device)
        one_hot_output[0][class_idx] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Get gradients and convert to positive
        gradients = self.gradients.data[0].cpu().numpy()
        positive_gradients = np.maximum(gradients, 0)
        # Get the output of the target layer
        target = self.conv_output.data[0].cpu().numpy()
        # Weight the target layer's output with the (positive) gradients
        weights = np.mean(positive_gradients, axis=(1, 2))
        cam = np.dot(target.transpose(1, 2, 0), weights).transpose(2, 0, 1)
        # Relu and normalize heatmap
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def apply_colormap_on_image(org_img, heatmap, alpha=0.4):
    # Resize heatmap to match the original image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).to(org_img.device).permute(2, 0, 1).float() / 255
    heatmap = F.interpolate(heatmap.unsqueeze(0), size=(org_img.size(1), org_img.size(2)), mode='bilinear',
                            align_corners=False).squeeze(0)
    # Apply heatmap on img
    superimposed_img = heatmap * alpha + org_img
    superimposed_img = superimposed_img / superimposed_img.max()
    return superimposed_img


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # Don't show axes for images

##########################


# predicting
import torch.nn.functional as F
# model
model = VGG()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the model
model.load_state_dict(torch.load(f"vgg_model_{epochs}.pth"))

#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# criterion = nn.CrossEntropyLoss()  # Assuming you're using CrossEntropyLoss for classification
#
#
#
#
#
#
# # Function to display an image
# def imshow(img):
#     img = img.cpu().numpy().transpose((1, 2, 0))  # Convert from tensor and reorder dimensions
#     mean = np.array([0.485, 0.456, 0.406])  # These values should match your normalization
#     std = np.array([0.229, 0.224, 0.225])  # These values should match your normalization
#     img = std * img + mean  # Unnormalize
#     img = np.clip(img, 0, 1)  # Clip to ensure [0,1] range
#     plt.imshow(img)
#     plt.show()
#

# Classes
#classes = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']  # Update this with your actual class names


##################
# Initialize GradCAM
# Assuming model is your VGG model and target_layer is the layer you're interested in
target_layer = model.features[-1]
grad_cam = SimpleGradCAM(model, target_layer)

# Select an image from your dataset
image, _ = next(iter(test_loader))
image.requires_grad_(True)

# Generate heatmap for a specific class index
class_idx = 0  # Example class index
heatmap = grad_cam.generate_heatmap(image.unsqueeze(0), class_idx)

# Apply the heatmap to the original image
superimposed_img = apply_colormap_on_image(image.squeeze(0), heatmap, alpha=0.4)

# Visualize
plt.imshow(superimposed_img.permute(1, 2, 0).cpu().numpy())
plt.show()




#######################
















#
# # Evaluation loop with visualization
# model.eval()
# total_loss = 0.0
# total_correct = 0
# total_images = 0
#
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         total_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total_correct += (predicted == labels).sum().item()
#         total_images += labels.size(0)
#
#         # Optional: Log images, predictions, and true labels to wandb
#         if total_images <= len(images):  # Log only the first batch
#             wandb_images = []
#             for i in range(min(len(images), 8)):  # Log up to 4 images per batch
#                 wandb_images.append(wandb.Image(
#                     images[i].cpu(),
#                     caption=f"True: {classes[labels[i]]} | Pred: {classes[predicted[i]]}"
#                 ))
#             wandb.log({"Test Examples": wandb_images})
#
# avg_loss = total_loss / len(test_loader)
# avg_accuracy = total_correct / total_images
#
# # Log the test loss and accuracy
# logging.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}")
# wandb.log({"Test Loss": avg_loss, "Test Accuracy": avg_accuracy})