import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim


class Trainer:
    def __init__(self, models, device, train_loader, criterion, optimizers):
        self.models = models
        self.device = device
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizers = optimizers

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for model_name, model in self.models.items():
                model.train()
                running_loss = 0.0
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizers[model_name].zero_grad()

                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizers[model_name].step()

                    running_loss += loss.item()

                print(
                    f"Epoch {epoch + 1}, Model {model_name}, Loss: {running_loss / len(self.train_loader)}"
                )


# Example usage
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define your models, optimizers, and criterion here
    MODELS = {
        "VGG19": models.vgg19(pretrained=True).to(DEVICE)
        # Add other models as needed
    }

    OPTIMIZERS = {
        "VGG19": optim.Adam(MODELS["VGG19"].parameters(), lr=0.001)
        # Add optimizers for other models as needed
    }

    CRITERION = nn.CrossEntropyLoss()

    # Assume TRAIN_LOADER is defined elsewhere
    TRAIN_LOADER = None  # Placeholder for the actual DataLoader

    trainer = Trainer(MODELS, DEVICE, TRAIN_LOADER, CRITERION, OPTIMIZERS)
    trainer.train(num_epochs=10)
