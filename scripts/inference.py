'''
File name: inference.py
Author: David Megli
Created: 2025-03-13
Description:
'''
import torch
from torchvision import transforms
from PIL import Image
from src.model import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP()
model.load_state_dict(torch.load("models/mnist_mlp.pth")) # Load the trained model
model.to(device)
model.eval()

transform = transforms.Compose([transforms.Greyscale(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

def predict(image_path):
    image = Image.open(image_path).convert("L") # Convert to greyscale, L stands for luminance
    image = transform(image).unsqueeze(0).to(device) # Add batch dimension and move to device

    with torch.no_grad():
        output = model(image)

    return output.argmax(1).item()

if __name__ == "__main__":
    image_path = "data/test.png"

    prediction = predict(image_path)
    print(f"Predicted digit: {prediction}")