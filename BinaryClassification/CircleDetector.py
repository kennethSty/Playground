import torch
from torch import nn
from typing import List

class CircleDetector(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: List[int]):
        super().__init__()

        layers = []

        # Input layer
        input_layer = nn.Linear(input_dim, hidden_layers[0])
        input_layer_act = nn.LeakyReLU(negative_slope = 0.01)

        layers.append(input_layer)
        layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(input_layer_act)

        # Hidden layers
        n_hidden_layers = len(hidden_layers)
        for i in range(1, n_hidden_layers):
            h_layer = nn.Linear(hidden_layers[i - 1], hidden_layers[i])
            h_act = nn.LeakyReLU(negative_slope = 0.01)
            layers.append(h_layer)
            layers.append(nn.BatchNorm1d(hidden_layers[i]))
            layers.append(h_act)

        # Output layer
        output_layer = nn.Linear(hidden_layers[-1], output_dim)
        output_act = nn.Sigmoid()
        layers.append(output_layer)
        layers.append(output_act)
        # Register all layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[3](x)
        for i in range(2, len(self.layers)-3):
            skipper = x
            new_val = self.layers[i](x)
            x = skipper + new_val
        x = self.layers[-3](x)    
        x = self.layers[-2](x)
        output = self.layers[-1](x)
        return output

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # Use kaiming for ReLU
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
