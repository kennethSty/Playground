from typing import List
import torch
from torch import nn

class NN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_nodes: List[int]):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_nodes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_nodes) - 1):
            layers.append(nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_nodes[-1], output_dim))

        # Create sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.layers(x)
        return logits
