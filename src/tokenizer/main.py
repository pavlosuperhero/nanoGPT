import torch

class Tokenizer:
    def __init__(self, corpus: str):
        self.chars = sorted(list(set(corpus)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, indices: list[int]) -> str:
        return ''.join([self.itos[i] for i in indices])

    def get_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)

    def print_tensor(self, data: torch.Tensor):
        print(f"Vocab size: {self.vocab_size}")
        print(f"Data tensor shape: {data.shape}")
        print(f"Data tensor dtype: {data.dtype}")
        print(f"First 10 tokens: {data[:10].tolist()}")

