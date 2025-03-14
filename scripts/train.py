import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb

def train_epoch(model, train_dataloader, optimizer, epoch="Unknown", device):
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
        
def val_epoch(model, val_dataloader, device):
    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad(): # no gradient computation is needed
        for (data, label) in tqdm(val_dataloader, desc=f"Evaluating", leave=False):
            data = data.to(device)
            label = label.to(device)
            
            logits = model(data)
            loss = F.cross_entropy(logits, label)

            val_loss += loss.item()
            correct += (logits.argmax(1) == label).sum().item()

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = correct / len(val_dataloader.dataset)
        return val_loss, val_accuracy

def train(model, train_dataloader, val_dataloader, optimizer, device, epochs=10, use_wandb=False):
    writer = SummaryWriter(log_dir="logs/")

    if use_wandb:
        wandb.init(project="mnist-mlp", config={"epochs": epochs})

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, train_dataloader, optimizer, epoch, device)
        val_loss, val_accuracy = val_epoch(model, val_dataloader, device)

        printf(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        if use_wandb:
            wandb.log({"Train Loss": train_loss, "Train Accuracy": train_accuracy, "Val Loss": val_loss, "Val Accuracy": val_accuracy})

    writer.close()
    if use_wandb:
        wandb.finish()
