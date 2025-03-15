'''
File name: model.py
Author: David Megli
Created: 2025-03-13
Description: MLP model implementation
'''
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers) # * unpacks the list

    def forward(self, x):
        #return self.layers(x.flatten(start_dim=1))
        return self.layers(x.view(x.size(0), -1))
    
class ResidualBlock(nn.Module):
    def __init__(self, width, num_layers=2):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(*[nn.Linear(width,width) if i%2==0 else nn.ReLU() for i in range(num_layers)])
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.layers(x))

class ResidualMLP(nn.Module):
    def __init__(self, layer_sizes, block_depth=2):
        super(ResidualMLP, self).__init__()
        self.input_layer = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.residual_blocks = nn.Sequential(*[ResidualBlock(layer_sizes[i], block_depth) for i in range(1, len(layer_sizes)-1, block_depth)])
        self.output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return self.softmax(x)