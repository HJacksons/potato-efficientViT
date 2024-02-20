from configurations import *
import torch
import logging
import wandb
from dataset import Dataset

# import pdb; pdb.set_trace()


class Trainer:
    def __init__(
        self, models, device, train_loader, vali_loader, criterion, optimizers
    ):
        self.models = models
        self.device = device
        self.train_loader = train_loader
        self.vali_loader = vali_loader
        self.criterion = criterion
        self.optimizer = optimizers
        self.best_acc = {model_name: 0.0 for model_name in self.models.keys()}

    # Train the model get loss and accuracy
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for model_name, model in self.models.items():
                model.train()
                running_loss, running_acc = 0.0, 0.0
                total_samples = 0
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.optimizer[model_name].zero_grad()

                    outputs = model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer[model_name].step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    running_acc += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

                avg_loss = running_loss / len(self.train_loader)
                avg_acc = running_acc / total_samples
                logging.info(
                    f"Epoch {epoch + 1}, Model {model_name}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}"
                )
                wandb.log(
                    {
                        f"{model_name} Train Loss": avg_loss,
                        f"{model_name} Train Accuracy": avg_acc,
                    }
                )

    # Validate the model get loss and accuracy
    def validate(self):
        for model_name, model in self.models.items():
            model.eval()
            running_loss, running_acc = 0.0, 0.0
            total_samples = 0
            with torch.no_grad():
                for images, labels in self.vali_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = model(images)
                    loss = self.criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    running_acc += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            avg_loss = running_loss / len(self.vali_loader)
            avg_acc = running_acc / total_samples
            logging.info(
                f"Model {model_name}, Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_acc:.4f}"
            )
            wandb.log(
                {
                    "Model": model_name,
                    "Validation Loss": avg_loss,
                    "Validation Accuracy": avg_acc,
                }
            )

            # Save the model if it has better accuracy than previously seen
            if avg_acc > self.best_acc[model_name]:
                self.best_acc[model_name] = avg_acc
                torch.save(model.state_dict(), f"{model_name}_best_model.pth")
                logging.info(
                    f"Saved new best model for {model_name} with accuracy: {avg_acc}"
                )


if __name__ == "__main__":
    dataset = Dataset()
    train_loader, vali_loader, _ = dataset.prepare_dataset()
    trainer = Trainer(MODELS, DEVICE, train_loader, vali_loader, CRITERION, OPTIMIZERS)
    trainer.train(num_epochs=EPOCHS)
    trainer.validate()

    wandb.finish()
