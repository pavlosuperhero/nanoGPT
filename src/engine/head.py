import torch
from torch import nn
from torch.nn import functional as F

class Head(nn.Module):

    def __init__(self, head_size, n_embd: int = 32, block_size: int = 8):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        v = self.value(x)
        out = weights @ v
        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd=n_embd, head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
