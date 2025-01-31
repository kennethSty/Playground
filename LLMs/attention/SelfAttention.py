import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W_query = nn.Linear(dim_in, dim_out, bias = False)
        self.W_key =  nn.Linear(dim_in, dim_out, bias = False)
        self.W_value =  nn.Linear(dim_in, dim_out, bias = False)

    def forward(self, input_ids):
        quueries = input_ids @ W_query
        keys = input_ids @ W_key
        values = input_ids @ W_values

        attention_scores = quueries @ keys.T #make dot prod of embeddings per word
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, #normalize to avoid vanishing gradients 
            dim = -1 #apply s.t. probabs in row sum t 1
        )
