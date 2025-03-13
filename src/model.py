import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLUI())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #return self.layers(x.flatten(start_dim=1))
        return self.layers(x.view(x.size(0), -1))