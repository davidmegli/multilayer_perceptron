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
def get_mnist_dataloaders(batch_size=64, num_workers=4):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    val_size = 5000
    indexes = np.random.permutation(len(train_set))
    val_set = Subset(train_set, indexes[:val_size])
    train_set = Subset(train_set, indexes[val_size:])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader, val_dataloader