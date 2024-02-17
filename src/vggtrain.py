from vgg import VGG
import torch
from torch import nn
from dataset import Dataset
import logging
import os
import wandb


# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up wandb
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

# We are going to train the vgg on our dataset

# Load the dataset
dataset = Dataset()
train_loader, vali_loader, test_loader = dataset.prepare_dataset()

# Model
model = VGG()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Train the model, log train loss, train accuracy, validation loss, validation accuracy
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_correct / total_samples
    logging.info(
        f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}"
    )
    wandb.log({"Train Loss": avg_loss, "Train Accuracy": avg_accuracy})

    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in vali_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(vali_loader)
    avg_accuracy = total_correct / total_samples
    logging.info(
        f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}"
    )
    wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": avg_accuracy})

# Save the model
torch.save(model.state_dict(), f"vgg_model_{epochs}.pth")
wandb.save(f"vgg_model_{epochs}.pth")
logging.info("Model saved.")


# predicting
import torch.nn.functional as F
# model
model = VGG()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the model
model.load_state_dict(torch.load(f"vgg_model_{epochs}.pth"))
model.eval()

# Load the test data
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Predict the first image in the batch
with torch.no_grad():
    outputs = model(images[0].unsqueeze(0))
    _, predicted = torch.max(outputs, 1)

# Print the predicted class
logging.info(f"Predicted class: {predicted.item()}", "Actual class: {labels[0].item()}")

