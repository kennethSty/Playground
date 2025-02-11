import torch
from CausalAttention import CausalAttention
from SeqMultiHeadAttention import SeqMultiHeadAttention
from ParMultiHeadAttention import ParMultiHeadAttention

def main():
    ca = CausalAttention(d_in = 3, d_out = 6, context_length = 3, dropout = 0.0) 
    smca = SeqMultiHeadAttention(d_in = 3, d_out = 6, context_length = 3, num_heads = 2, dropout = 0.0)
    pmca = ParMultiHeadAttention(d_in = 3, d_out = 6, context_length = 3, num_heads = 2, dropout = 0.0)
    
    x_1 = torch.Tensor([1.0, 2.0, 3.0]).unsqueeze(0)
    x_2 = torch.Tensor([2.0, 4.0, 6.0]).unsqueeze(0)
    x_3 = torch.randn(1, 3)
    x = torch.stack([x_1, x_2, x_3], dim = 1)

    z = ca(x)
    zsm = smca(x)
    zpm = pmca(x)
     
    print("Single Attention", z)
    print("Sequential Multi  Attention", zsm)
    print("Parallel Multi Attention", zpm)

if __name__ == "__main__":
    main()
