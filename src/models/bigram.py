import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, torch_manual_seed: int = 1337):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        torch.manual_seed(torch_manual_seed)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def print_model(self, xb, yb):
        logits, loss = self(xb, yb)
        self.logger.debug(f"Logits shape: {logits.shape}")
        self.logger.debug(f"Loss value: {loss.item()}")
