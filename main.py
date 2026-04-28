import torch
from sys import argv
from sys import exit
import torch.nn as nn
from torch.nn import functional as F

batch_size = 4
block_size = 8

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
    
def main(file_path: str = ''):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    tokenizer = Tokenizer(raw_text)

    data_tensor = tokenizer.get_tensor(raw_text)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Data tensor shape: {data_tensor.shape}")
    print(f"First 10 tokens: {data_tensor[:10].tolist()}")

if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python main.py <filepath>")
        exit(1)
        
    target_path = argv[1]
    main(target_path)
