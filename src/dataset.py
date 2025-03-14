'''
File name: dataset.py
Author: David Megli
Created: 2025-03-13
Description: MNIST dataloader function
'''
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
def get_mnist_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    val_size = 5000
    indexes = np.random.permutation(len[train_set])
    val_set = Subset(train_set, indexes[:val_set])
    train_set = Subset(train_set, indexes[val_set:])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_set, test_set, val_set