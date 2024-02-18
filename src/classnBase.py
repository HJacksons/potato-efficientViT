import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
import wandb
import os
import matplotlib.pyplot as plt
import io

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageClassnBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]  # Combine losses
        epoch_loss = torch.stack(batch_losses).mean()  # Aggregate losses
        batch_accs = [x['val_acc'] for x in outputs]  # Combine accuracies
        epoch_acc = torch.stack(batch_accs).mean()  # Aggregate accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['train_loss'],
                                                                                         result['val_loss'],
                                                                                         result['val_acc']))
        wandb.log({"Train Loss": result['Train Accuracy'], "Validation Loss": result['val_loss'], "Validation Accuracy": result['val_acc']})






