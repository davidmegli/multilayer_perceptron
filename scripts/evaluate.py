'''
File name: evaluate.py
Author: David Megli
Created: 2025-03-13
Description:
'''
import torch
from torch import nn
from src.dataset import get_mnist_dataloaders
from src.model import MLP
import src.config as config
from src.train_utils import evaluate
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(config.LAYER_SIZES).to(device)
model.load_state_dict(torch.load("models/mnist_mlp.pth")) # Load the trained model
model.eval()

train_dataloader, test_dataloader, val_dataloader = get_mnist_dataloaders(config.BATCH_SIZE, config.NUM_WORKERS)

test_loss, test_accuracy = evaluate(model, test_dataloader, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")