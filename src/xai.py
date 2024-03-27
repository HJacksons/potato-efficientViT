from configurations import *
from models import ViT
import shap
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import Dataset
# import wandb
from PIL import Image
import io


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, pixel_values):
        logits, _ = self.model(pixel_values)
        return logits


def main():
    model = ViT()
    model.load_state_dict(torch.load('HybridModel_model_potatodata_CV_False.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Wrap your model
    wrapped_model = ModelWrapper(model)

    ds = Dataset()
    _, _, test_loader = ds.prepare_dataset()

    # When running on the test dataset
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = wrapped_model(images)

        # Convert logits to probabilities
        # probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

        # Select a subset of images for SHAP
        subset_indices = np.random.choice(np.arange(images.shape[0]), size=4, replace=False)
        subset_images = images[subset_indices]

        # Use SHAP to explain test predictions
        e = shap.GradientExplainer(wrapped_model, subset_images)
        shap_values = e.shap_values(subset_images)

        # Plot the SHAP values
        shap.image_plot(shap_values, -subset_images.detach().cpu().numpy())
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image.show()
        wandb.log({"SHAP Explainer": [wandb.Image(image)]})

    # plt.show()


if __name__ == '__main__':
    main()
