from torch import nn
from Typing import List

class BasicModel(nn.Module):
    def __init__(self, output_dim, input_channels: int, input_height:int, input_width: int, hidden_units: int):
        super().__init__()
        elems_after_flatten = input_channels * input_width * input_height
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = elems_after_flatten, out_features = hidden_units),
            nn.Linear(in_features = hidden_units, out_features = hidden_units)
            nn.Linear(in_features = hidden_units, out_features = output_dim)
        )

    def forward(self, x)
    return self.layer_stack(x)
        
        
