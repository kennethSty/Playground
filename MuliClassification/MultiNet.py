from torch import nn

class MultiNet(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_nodes: int):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_features, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, output_features)
        )
        

    def forward(self, x):
        return self.linear_stack(x)
