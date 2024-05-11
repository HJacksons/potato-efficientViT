import matplotlib.pyplot as plt
from PIL import Image
import io
from configurations import *
from dataset import Dataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)
import seaborn as sns
import torch
import numpy as np
import logging


class Tester:
    def __init__(self, models, device, test_loader, criterion):
        self.device = device
        self.models = {
            model_name: self.load_model(model_class, SAVED_MODELS[model_name])
            for model_name, model_class in models.items()
        }
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
            class_correct = list(0.0 for i in range(len(CLASSES)))
            class_total = list(0.0 for i in range(len(CLASSES)))
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
                    class_correct, class_total = self.calculate_classwise_accuracy(
                        labels, predicted, class_correct, class_total
                    )

            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)
            avg_loss = running_loss / len(self.test_loader)
            avg_acc = running_acc / total_samples
            self.calculate_and_log_metrics(
                model_name, all_labels, all_predictions, avg_loss, class_correct, class_total
            )
            self.log_confusion_matrix(model_name, all_labels, all_predictions)

    @staticmethod
    def calculate_classwise_accuracy(labels, predicted, class_correct, class_total):
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                class_correct[label] += 1
            class_total[label] += 1
        return class_correct, class_total

    @staticmethod
    def calculate_and_log_metrics(model_name, all_labels, all_predictions, avg_loss, class_correct, class_total):
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average="macro")
        recall = recall_score(all_labels, all_predictions, average="macro")
        f1 = f1_score(all_labels, all_predictions, average="macro")
        mcc = matthews_corrcoef(all_labels, all_predictions)
        logging.info(
            f"Model {model_name}, Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}"
        )
        # Class-wise metrics
        precision_classwise = precision_score(all_labels, all_predictions, average=None)
        recall_classwise = recall_score(all_labels, all_predictions, average=None)
        f1_classwise = f1_score(all_labels, all_predictions, average=None)

        for i in range(len(CLASSES)):
            class_accuracy = 100 * class_correct[i] / class_total[i]
            mcc_classwise = matthews_corrcoef(all_labels == i, all_predictions == i)
            logging.info(
                f"Class {CLASSES[i]}, Accuracy: {class_accuracy:.2f}%, Precision: {precision_classwise[i]:.4f}, Recall: {recall_classwise[i]:.4f}, F1: {f1_classwise[i]:.4f}, MCC: {mcc_classwise:.4f}"
            )

    @staticmethod
    def log_confusion_matrix(model_name, all_labels, all_predictions):
        cm = confusion_matrix(all_labels, all_predictions, labels=range(len(CLASSES)))
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
        )
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({f"{model_name} Confusion Matrix": wandb.Image(image)})
        plt.close()


if __name__ == "__main__":
    dataset = Dataset()
    _, _, test_loader = dataset.prepare_dataset()
    tester = Tester(MODELS, DEVICE, test_loader, CRITERION)
    tester.test_models()
    wandb.finish()
