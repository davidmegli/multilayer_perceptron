'''
File name: evaluate.py
Author: David Megli
Created: 2025-03-13
Description:
'''
import torch
from torch import nn
from src.dataset import get_mnist_dataloaders
from src.models import MLP
import src.config as config
from src.train_utils import evaluate
import numpy as np
import os

def evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(config.LAYER_SIZES)
    # Find the name of the last saved model in the models directory
    model_dir = "models"
    last_model = model_dir + "\\" + sorted(os.listdir(model_dir))[-1]
    model.load_state_dict(torch.load(last_model)) # Load the trained model
    model.to(device)
    model.eval()

    print(f"Model loaded from {last_model}")

    train_dataloader, test_dataloader, val_dataloader = get_mnist_dataloaders(config.BATCH_SIZE, config.NUM_WORKERS)
    
    test_loss, test_accuracy = evaluate(model, test_dataloader, device)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    evaluation()