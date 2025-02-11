import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, 
                dropout: float, qkv_bias = False):
        super().__init__()
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length),
            diagonal = 1) #values above diagonal = 1, rest 0
        ) # register buffer set of "params that are not trainable" 

    def forward(self, x: torch.Tensor): # x_dim = (context, d_in) 
        batch_size, context_size, d_in = x.shape # best practise to show dimension assumption

        queries = self.W_query(x) 
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1,2) # query index is row index and key index is col index
        attention_scores.masked_fill( #set values above diagonal to infinity to they will be set tp 0 in softmax
            self.mask.bool(), -torch.inf
        ) 

        normalized_scores = attention_scores / self.d_out ** 0.5 
        attention_weights = torch.softmax(normalized_scores, dim = -1) #on erow (i.e. values across col indices) should sum to 1
        attention_weights = self.dropout(attention_weights)

        #multiply attention weights from left to get lin comb of value rows with attention weights as coefs 
        context_vectors = attention_weights @ values 

        return context_vectors
