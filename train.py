'''
File name: train.py
Author: David Megli
Created: 2025-03-14
Description: Training and evaluation functions for PyTorch models
'''
import torch
from torch import optim
from src.dataset import get_mnist_dataloaders
from src.model import MLP
from src.train_utils import train
import src.config as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader, test_dataloader, val_dataloader = get_mnist_dataloaders(config.BATCH_SIZE, config.NUM_WORKERS)

model = MLP(config.LAYER_SIZES).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.LR)
# criterion = torch.nn.CrossEntropyLoss()

USE_WANDB = False
if __name__ == "__main__":
    train(model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        val_dataloader=val_dataloader,
        device=device,
        epochs=config.EPOCHS,
        use_wandb=USE_WANDB)
