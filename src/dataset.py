import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
def get_mnist_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    val_size = 5000
    indexes = np.random.permutation(len[train_set])
    val_set = Subset(train_set, indexes[:val_set])
    train_set = Subset(train_set, indexes[val_set:])
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_set, test_set, val_set