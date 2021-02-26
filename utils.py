import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset="cifar", transform=None, valid_size=None, batches=128):

    if transform is None:
        transform = transforms.ToTensor()

    if dataset == "cifar":
        train_data = datasets.CIFAR10("./cifar/train", train=True, 
                                      transform=transform, download=True)
    
        test_data = datasets.CIFAR10("./cifar/test", train=False, 
                                     transform=transform, download=True)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batches)

    if valid_size is None:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batches, shuffle=True)   

        return (train_loader, test_loader)
    else:
        indices = np.arange(0, len(train_data))
        np.random.shuffle(indices)

        train_samples = int(valid_size * len(train_data))

        train_ind = indices[:train_samples]
        valid_ind = indices[train_samples:]

        train_samp = torch.utils.data.SequentialSampler(train_ind)
        valid_samp = torch.utils.data.SequentialSampler(valid_ind)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batches, sampler=train_samp)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batches, sampler=valid_samp)

        return (train_loader, valid_loader, test_loader)


def train(model, optimizer, criterion, data_loader):
    device = get_device()

    model.to(device)
    model.train()

    epoch_loss = 0

    for data, target in data_loader:        
        data, target = data.to(device), target.to(device)

        out = model(data)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)


def validate(model, criterion, data_loader):
    device = get_device()

    model.to(device)
    model.eval()

    validation_loss = 0

    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            out = model(data)
            loss = criterion(out, target)

            validation_loss += loss.item()
  
    return validation_loss / len(data_loader)


def evaluate_acc(model, data_loader):
    device = get_device()

    model.to(device)
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            out = model(data)

            _, y_hat = torch.topk(out, k=1, dim=1)
            y_hat = y_hat.reshape(*target.shape)

            total += y_hat.size(0)
            correct += (y_hat == target).sum().item()
    
    return correct / total


def plot_learning_curves(train_losses, validation_losses, epoch_step=5):
    x_values = epoch_step * np.arange(1, len(train_losses) + 1, 1)

    plt.figure(figsize=(8,8))
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.plot(x_values, train_losses)
    plt.plot(x_values, validation_losses)
    plt.gca().legend(('Train', 'Validation'), loc="upper left", fontsize=15)
    plt.show()