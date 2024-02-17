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
        self.handlers = []  # Handlers for the hooks
        self.feature_maps = None  # To store the feature maps
        self.gradients = None  # To store the gradients

        # Register hook to the feature layer
        self.handlers.append(feature_layer.register_forward_hook(self.save_feature_maps))
        # Ensure gradients are saved
        self.handlers.append(feature_layer.register_full_backward_hook(self.save_gradients))

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output
        self.feature_maps.retain_grad()  # Retain gradients for non-leaf tensors

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # grad_output[0] contains gradients with respect to the output

    def generate_heatmap(self, input_image, class_idx):
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0]  # Handle models that return a tuple

        self.model.zero_grad()
        class_score = output[:, class_idx].sum()
        class_score.backward(retain_graph=True)

        if self.gradients is not None and self.feature_maps is not None:
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)

            # Apply the gradients onto the feature map
            weighted_feature_maps = self.feature_maps[0] * pooled_gradients

            # Generate the heatmap
            heatmap = torch.mean(weighted_feature_maps, dim=0)
            heatmap = torch.clamp(heatmap, min=0)
            heatmap /= torch.max(heatmap)

            return heatmap.cpu().data.numpy()
        else:
            raise RuntimeError("Gradients or feature maps are not populated.")

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()



def apply_colormap_on_image(org_img, heatmap, alpha=0.6, colormap=plt.cm.jet):
    # Resize heatmap to match the image size
    heatmap = np.uint8(255 * heatmap)  # Convert to 8-bit int
    heatmap = colormap(heatmap)[:, :, :3]  # Apply colormap and remove alpha channel
    heatmap = torch.from_numpy(heatmap).to(device).float() / 255
    with torch.no_grad():
        cam = heatmap + alpha * org_img
        cam = cam / cam.max()  # Normalize
    return cam

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
grad_cam = GradCAM(model, model.model.features[-1])

# Visualization for a single image from the test set
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

# Select an image and predict
image_idx = 0  # Change as needed
image_tensor = images[image_idx].unsqueeze(0)  # Add batch dimension
output = model(image_tensor)
_, predicted_class = torch.max(output, 1)
predicted_class = predicted_class.item()

# Generate Grad-CAM heatmap
heatmap = grad_cam.generate_heatmap(image_tensor, predicted_class)

# Unnormalize the image for display
img_display = images[image_idx] * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
img_display = img_display.cpu()

# Apply heatmap on original image
cam_img = apply_colormap_on_image(img_display, heatmap)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
imshow(torchvision.utils.make_grid(image_tensor.cpu().data, normalize=True))
plt.title('Original Image')

plt.subplot(1, 2, 2)
imshow(torchvision.utils.make_grid(cam_img.cpu().data, normalize=True))
plt.title('Grad-CAM')

plt.show()

# Clean up hooks
grad_cam.remove_hooks()

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