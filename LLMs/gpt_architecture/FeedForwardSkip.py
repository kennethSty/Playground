import torch
import torch.nn as  nn
from typing import List

class FeedForwardSkip(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden_nodes: List[int], use_skip_connection: bool):
        layer_list = []
        layer_list.append(
            nn.Sequential(
                nn.Linear(dim_in, hidden_nodes[0]),
                nn.ReLU()
            )
        )
    
        for i in range(len(hidden_nodes) - 1):
            layer_list.append(
                nn.Sequential(
                    nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]),
                    nn.ReLU()
                )
            )

        layers.append(nn.Linear(hidden_nodes[i], dim_out))
        self.layers = nn.ModuleList(layer_list)
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_skip_connection and x.shape == layer_output.shape:
                x = layer_output + x
            else:
                x = layer_output

        return x
