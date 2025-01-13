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
        input_norm = nn.BatchNorm1d(hidden_layers[0])
        input_layer_act = nn.ReLU()

        layers.append(input_layer)
        layers.append(input_norm)
        layers.append(input_layer_act)

        # Hidden layers
        n_hidden_layers = len(hidden_layers)
        for i in range(1, n_hidden_layers):
            h_layer = nn.Linear(hidden_layers[i - 1], hidden_layers[i])
            h_norm = nn.BatchNorm1d(hidden_layers[i])
            h_act = nn.ReLU()
            layers.append(h_layer)
            layers.append(h_norm)
            layers.append(h_act)

        # Output layer
        output_layer = nn.Linear(hidden_layers[-1], output_dim)
        output_act = nn.Sigmoid()
        layers.append(output_layer)
        layers.append(output_act)
        # Register all layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
