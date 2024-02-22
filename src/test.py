import matplotlib.pyplot as plt
import io
from configurations import *
from dataset import Dataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import seaborn as sns
import torch
import numpy as np
import logging


class Tester:
    def __init__(self, models, device, test_loader, criterion):
        self.models = {
            model_name: self.load_model(model_class, SAVED_MODELS[model_name])
            for model_name, model_class in models.items()
        }
        self.device = device
        self.test_loader = test_loader
        self.criterion = criterion

    def load_model(self, model_class, saved_model_path):
        model = model_class().to(self.device)
        model.load_state_dict(torch.load(saved_model_path, map_location=self.device))
        model.eval()
        return model

    def test_models(self):
        for model_name, model in self.models.items():
            model.eval()
            running_loss, running_acc = 0.0, 0.0
            total_samples = 0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    # Caters for VIT model
                    if isinstance(outputs, tuple):
                        logits, loss = outputs
                        if loss is None:
                            loss = self.criterion(logits, labels)
                    else:
                        logits = outputs
                        loss = self.criterion(logits, labels)
                    # End catering for VIT model
                    running_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    running_acc += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)

            avg_loss = running_loss / len(self.test_loader)
            avg_acc = (
                running_acc / total_samples
            )  # WIll report accuracy from sklearn instead of this
            accuracy = accuracy_score(
                all_labels, all_predictions
            )  # A = (TP + TN) / (TP + TN + FP + FN)
            precision = precision_score(
                all_labels, all_predictions, average="macro"
            )  # P = TP / (TP + FP)
            recall = recall_score(
                all_labels, all_predictions, average="macro"
            )  # R = TP / (TP + FN)
            f1 = f1_score(
                all_labels, all_predictions, average="macro"
            )  # F1 = 2 * (P * R) / (P + R)

            logging.info(
                f"Model {model_name}, Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )
            wandb.log(
                {
                    f"{model_name} Test Loss": avg_loss,
                    f"{model_name} Test Accuracy": accuracy,
                    f"{model_name} Precision": precision,
                    f"{model_name} Recall": recall,
                    f"{model_name} F1": f1,
                }
            )
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_predictions)
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"{model_name} Confusion Matrix")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            wandb.log({f"{model_name} Confusion Matrix": wandb.Image(buf)})
            plt.close()


if __name__ == "__main__":
    dataset = Dataset()
    _, _, test_loader = dataset.prepare_dataset()
    tester = Tester(MODELS, DEVICE, test_loader, CRITERION)
    tester.test_models()
    wandb.finish()
