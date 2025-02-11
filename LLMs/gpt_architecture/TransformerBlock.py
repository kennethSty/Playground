import torch
import torch.nn as nn
from LayerNorm import LayerNorm
from typing import Dict
from ParMultiHeadAttention import ParMultiHeadAttention
from FeedForward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.multi_head_attn = ParMultiHeadAttention(
            d_in = config["emb_dim"],
            d_out = config["emb_dim"],
            context_length = config["context_length"],
            dropout = config["drop_rate"],
            num_heads = config["n_heads"]
            )
        self.layer_norm1 = LayerNorm(emb_dim = config["emb_dim"])
        self.layer_norm2 = LayerNorm(emb_dim = config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        batch_size, n_tokens, emb_dim = x.shape

        x_norm = self.layer_norm1(x)
        z = self.multi_head_attn(x_norm)
        z = self.dropout(z)
        x = z + x #skip connection

        x_norm = self.layer_norm2(x)
        z = self.feed_forward(x_norm)
        z = self.dropout(z)
        x = z + x #skip connection
        
        return x
