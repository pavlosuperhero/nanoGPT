import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size: int = 8, dropout: float = 0.2):
        super().__init__()
        self.n_head = num_heads
        self.n_embd = n_embd
        self.dropout_p = dropout

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()

        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout_p if self.training else 0.0, 
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.dropout(self.proj(out))
        return out
