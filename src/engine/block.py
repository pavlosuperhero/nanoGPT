from torch import nn

from .feedforward import FeedForward
from .head import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, n_head, n_embd: int = 32, block_size: int = 8, dropout: float = 0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd=n_embd, num_heads=n_head, head_size=head_size, block_size=block_size, dropout=dropout)
        self.ffwd = FeedForward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

        
