import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from src.engine.block import Block
from src.engine.head import MultiHeadAttention
from src.engine.feedforward import FeedForward

class BigramLanguageModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 torch_manual_seed: int = 1337,
                 n_embd: int = 32,
                 block_size: int = 8,
                 n_head: int = 4,
                 n_layer: int = 6,
                 dropout: float = 0.2):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        torch.manual_seed(torch_manual_seed)
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.possition_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
#        self.blocks = nn.Sequential(
#            Block(n_embd=n_embd, n_head=4, block_size=block_size),
#            Block(n_embd=n_embd, n_head=4, block_size=block_size),
#            Block(n_embd=n_embd, n_head=4, block_size=block_size),
#            nn.LayerNorm(n_embd)
#        )
#        self.sa_head = MultiHeadAttention(4,n_embd//4)
#        self.ffwd = FeedForward(n_embd)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.possition_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def print_model(self, xb, yb):
        logits, loss = self(xb, yb)
        self.logger.debug(f"Logits shape: {logits.shape}")
        self.logger.debug(f"Loss value: {loss.item()}")
