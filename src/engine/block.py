from torch import nn

from .feedforward import FeedForward
from .head import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, n_head, n_embd: int = 32):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd=n_embd, num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x

        
