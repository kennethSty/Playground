from torch import nn
import numpy as np

class RegNet(nn.Module):
    def __init__(self, input_dim, output_dim, nodes_list):
        super().__init__()

        #Create a Listt of Layers. 
        layers = []
        number_dense_layers = len(nodes_list)
        
        for i in range(number_dense_layers):
        
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = nodes_list[i-1]

            layer = nn.Linear(in_dim, nodes_list[i])
            layer_batch_norm = nn.BatchNorm1d(nodes_list[i])
            layer_activation = nn.ReLU()
            layer_dropout = nn.Dropout(p = np.random.uniform(0, 0.5))

            layers.append(layer)
            layers.append(layer_batch_norm)
            layers.append(layer_activation)
            layers.append(layer_dropout)

        output_layer = nn.Linear(nodes_list[-1], output_dim)
        layers.append(output_layer)
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
        return x
