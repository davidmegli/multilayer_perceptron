'''
File name: train_utils.py
Author: David Megli
Created: 2025-03-13
Description:
'''

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb
import os

def save_checkpoint(model, epoch, val_loss, val_acc, path="models/"):
    os.makedirs(path, exist_ok=True) # create the directory if it doesn't exist
    model_path = os.path.join(path, f"model_epoch_{epoch+1}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

def train_epoch(model, train_dataloader, optimizer, epoch="Unknown", device="cpu"):
    model.train() # set model to training mode
    train_loss, correct = 0, 0 # initialize loss and number of correct predictions
    for (data, label) in tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=True):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        
        logits = model(data) # the logits are the output of the model, it's a vector of raw predictions, in the case of a classification of a batch of images, it's a tensor of shape (batch_size, num_classes)
        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (logits.argmax(1) == label).sum().item()

    train_loss /= len(train_dataloader.dataset)
    train_accuracy = correct / len(train_dataloader.dataset)
    return train_loss, train_accuracy
        
def evaluate(model, dataloader, device):
    model.eval()
    loss, correct = 0, 0

    with torch.no_grad(): # no gradient computation is needed
        for (data, label) in tqdm(dataloader, desc=f"Evaluating", leave=False):
            data = data.to(device)
            label = label.to(device)
            
            logits = model(data)
            loss = F.cross_entropy(logits, label)

            loss += loss.item()
            correct += (logits.argmax(1) == label).sum().item()

        loss /= len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        return loss, accuracy

def train(model, train_dataloader, val_dataloader, optimizer, device, epochs=10, use_wandb=False):
    writer = SummaryWriter(log_dir="logs/")

    if use_wandb:
        wandb.init(project="mnist-mlp", config={"epochs": epochs})

    best_val_accuracy = 0

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, epoch, device)
        val_loss, val_accuracy = evaluate(model, val_dataloader, device)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        if use_wandb:
            wandb.log({"Train Loss": train_loss, "Train Accuracy": train_accuracy, "Val Loss": val_loss, "Val Accuracy": val_accuracy})

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model, epoch, val_loss, val_accuracy)

    writer.close()
    if use_wandb:
        wandb.finish()
