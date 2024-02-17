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

class GradCAM:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.gradients = None
        self.feature_maps = None
        feature_layer.register_forward_hook(self.save_feature_maps)
        feature_layer.register_full_backward_hook(self.save_gradients)

    def save_gradients(self, *args):
        self.gradients = args[1]

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def generate_heatmap(self, input_image, class_idx):
        # Forward pass
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0]  # For models returning a tuple

        # Zero grads
        self.model.zero_grad()

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(device)
        one_hot_output[0][class_idx] = 1

        # Backward pass
        output.backward(gradient=one_hot_output, retain_graph=True)

        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the feature map with the gradients
        for i in range(pooled_gradients.size()[0]):
            self.feature_maps[0][i] *= pooled_gradients[i]

        # Average the channels of the feature maps
        heatmap = torch.mean(self.feature_maps, dim=1).squeeze().cpu()

        # Relu on top of the heatmap
        heatmap = np.maximum(heatmap, 0)

        # Normalize the heatmap
        heatmap /= torch.max(heatmap)

        return heatmap.numpy()


def apply_colormap_on_image(org_img, heatmap, alpha=0.6, colormap=plt.cm.jet):
    # Apply heatmap on image
    heatmap = np.uint8(255 * heatmap)
    heatmap = colormap(heatmap)
    heatmap = torch.from_numpy(heatmap).to(device).float() / 255
    heatmap = heatmap[:, :, :3]  # remove alpha channel
    with torch.no_grad():
        cam = heatmap + alpha * org_img
        cam = cam / cam.max()
    return cam

##########################


# predicting
import torch.nn.functional as F
# model
model = VGG()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the model
model.load_state_dict(torch.load(f"vgg_model_{epochs}.pth"))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()  # Assuming you're using CrossEntropyLoss for classification






# Function to display an image
def imshow(img):
    img = img.cpu().numpy().transpose((1, 2, 0))  # Convert from tensor and reorder dimensions
    mean = np.array([0.485, 0.456, 0.406])  # These values should match your normalization
    std = np.array([0.229, 0.224, 0.225])  # These values should match your normalization
    img = std * img + mean  # Unnormalize
    img = np.clip(img, 0, 1)  # Clip to ensure [0,1] range
    plt.imshow(img)
    plt.show()


# Classes
classes = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']  # Update this with your actual class names


##################
# Initialize GradCAM with the model and the last convolutional layer
#grad_cam = GradCAM(model, model.features[-1])
grad_cam = GradCAM(model, model.model.features[-1])


# Choose an image from your test set
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Assuming you're interested in the first image in the batch
image_idx = 0
image_tensor = images[image_idx].unsqueeze(0)  # Add batch dimension

# Predict class for the selected image
output = model(image_tensor)
_, predicted_class = torch.max(output, 1)
predicted_class = predicted_class.item()

# Generate heatmap
heatmap = grad_cam.generate_heatmap(image_tensor, predicted_class)

# Convert heatmap to a displayable format
# Note: Implement the apply_colormap_on_image function based on the GradCAM example provided
cam_image = apply_colormap_on_image(image_tensor.cpu().squeeze(), heatmap)  # Remove batch dimension and move to CPU

# Display original image and Grad-CAM heatmap overlay
imshow(images[image_idx].cpu())  # Original image
imshow(cam_image)  # Grad-CAM overlay

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