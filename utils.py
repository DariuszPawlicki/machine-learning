import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset="cifar10", transform=None, train_size=None, batch_size=128):

    if transform is None:
        transform = transforms.ToTensor()

    if dataset == "cifar10":
        train_data = datasets.CIFAR10("./cifar10/train", train=True, 
                                      transform=transform, download=True)
    
        test_data = datasets.CIFAR10("./cifar10/test", train=False, 
                                     transform=transform, download=True)
    elif dataset == "mnist":
        train_data = datasets.MNIST("./mnist/train", train=True, 
                                      transform=transform, download=True)
    
        test_data = datasets.MNIST("./mnist/test", train=False, 
                                     transform=transform, download=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    if train_size is None:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)   

        return (train_loader, test_loader)
    else:
        indices = np.arange(0, len(train_data))
        np.random.shuffle(indices)

        train_samples = int(train_size * len(train_data))

        train_ind = indices[:train_samples]
        valid_ind = indices[train_samples:]

        train_samp = torch.utils.data.SequentialSampler(train_ind)
        valid_samp = torch.utils.data.SequentialSampler(valid_ind)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_samp)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_samp)

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


def train_gan(model, discr_optim, gene_optim, criterion, data_loader, on_images):
    device = get_device()

    model.to(device)
    model.train()

    discr_epoch_loss = []
    gene_epoch_loss = []

    for real_data, _ in data_loader:
        
        # DISCRIMINATOR

        real_data = real_data.to(device)
        
        if on_images == True:
            noise_discr = torch.randn(real_data.shape[0], model.latent_dimension, 1, 1)
            noise_gene = torch.randn(real_data.shape[0], model.latent_dimension, 1, 1)
        else:
            noise_discr = torch.randn(real_data.shape[0], model.latent_dimension)
            noise_gene = torch.randn(real_data.shape[0], model.latent_dimension)
        
        noise_discr = noise_discr.to(device)
        noise_gene = noise_gene.to(device)

        generated_data = model(noise_discr, generator=True)

        discr_real_out = model(real_data).reshape(-1)
        discr_gene_out = model(generated_data).reshape(-1)

        discr_real_loss = criterion(discr_real_out, torch.ones_like(discr_real_out))
        discr_gene_loss = criterion(discr_gene_out, torch.zeros_like(discr_gene_out))

        discr_loss = discr_real_loss + discr_gene_loss
        
        discr_optim.zero_grad()
        discr_loss.backward()
        discr_optim.step()

        discr_epoch_loss.append(discr_loss.item())
        
        # GENERATOR

        generated_data = model(noise_gene, generator=True)
        
        discr_gene_out = model(generated_data).reshape(real_data.shape[0], 1)

        gene_loss = criterion(discr_gene_out, torch.ones_like(discr_gene_out))

        gene_optim.zero_grad()
        gene_loss.backward()
        gene_optim.step()
        
        gene_epoch_loss.append(gene_loss.item())
    
    return (np.mean(discr_epoch_loss), np.mean(gene_epoch_loss))


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