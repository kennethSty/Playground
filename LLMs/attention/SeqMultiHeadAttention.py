import torch 
import torch.nn as nn
from CausalAttention import CausalAttention

class SeqMultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, num_heads, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in = d_in, 
                d_out = d_out, 
                context_length = context_length,
                dropout = dropout,
                qkv_bias = qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape[0], x.shape[1], x.shape[2]
        head_outputs = [head(x) for head in self.heads]
        results = torch.concat(head_outputs, dim = -1)
        return results
