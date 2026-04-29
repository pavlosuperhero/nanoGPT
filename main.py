from sys import argv
from sys import exit

import torch
from src.data.tokenizer import Tokenizer
from src.data.datasets import BatchGenerator
from src.models.bigram import BigramLanguageModel

import logging
logging.basicConfig(level=logging.DEBUG)

batch_size = 4
block_size = 8

    
def main(file_path: str = ''):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    tokenizer = Tokenizer(raw_text)
    data_tensor = tokenizer.get_tensor(raw_text) 
    
    training = BatchGenerator(data=data_tensor, batch_size=batch_size, block_size=block_size)
    
    xb, yb = training.get_batch('train') 
    
    m = BigramLanguageModel(tokenizer.vocab_size)

    tokenizer.print_tensor(data_tensor)
    print('\n---\n')
    
    training.print_batch()
    print('\n---\n')
    
    m.print_model(xb, yb)
    print('\n---\n')
    
    print(tokenizer.decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))



if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python main.py <filepath>")
        exit(1)
        
    target_path = argv[1]
    main(target_path)
