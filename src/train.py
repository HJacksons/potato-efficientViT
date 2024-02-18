import torch
from CNNModel import classnModel
from dataset import Dataset
from utils import accuracy

# from classnBase import plot_accuracies, plot_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# Save checkpoint to disk when val_loss improves
def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)


# Load checkpoint from disk when resuming training
def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Early stopping when val_loss stops improving
def early_stopping(val_loss, best_loss, patience=3, counter=0):
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            return True, best_loss, counter
    return False, best_loss, counter


# Modify the fit function
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    best_loss = float('inf')
    stop_counter = 0
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # After each epoch, print memory usage
        print(f"Epoch {epoch}:")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GB")
        torch.cuda.reset_max_memory_allocated(device)

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        history.append(result)
        model.epoch_end(epoch, result)
        # Save checkpoint
        # save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
        #                f'model_epoch_{epoch}.pth')
        # Check for early stopping
        # stop, best_loss, stop_counter = early_stopping(result['val_loss'], best_loss, patience=3, counter=stop_counter)
        # if stop:
        #    print(f"Early stopping triggered at epoch {epoch}")
        #    break
    return history


def main():
    num_epochs = 50
    opt_func = torch.optim.Adam
    lr = 2e-5

    model = classnModel().to(device)
    dataset = Dataset()
    train_loader, vali_loader, test_loader = dataset.prepare_dataset()
    # fit the model
    history = fit(num_epochs, lr, model, train_loader, vali_loader, opt_func)

    torch.save(model.state_dict(), f'model100{num_epochs}.pth')

    return history


if __name__ == '__main__':
    history = main()
