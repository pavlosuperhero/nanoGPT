from sys import argv
from sys import exit

import torch
from src.data.tokenizer import Tokenizer
from src.data.datasets import BatchGenerator
from src.engine.trainer import Trainer
from src.models.bigram import BigramLanguageModel

import logging
logging.basicConfig(level=logging.DEBUG)

batch_size = 4
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate=1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200
    
def main(file_path: str = ''):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    tokenizer = Tokenizer(raw_text)
    data_tensor = tokenizer.get_tensor(raw_text) 
    
    dataset = BatchGenerator(data=data_tensor, batch_size=batch_size, block_size=block_size)
    
    xb, yb = dataset.get_batch('train') 
    
    model = BigramLanguageModel(tokenizer.vocab_size)
    m = model.to(device)
    trainer = Trainer(parameters=model.parameters(), learning_rate=learning_rate)
    trainer.count_loss(max_iters=max_iters, eval_interval=eval_interval, batch_generator=tokenizer, model=model, eval_iters=eval_iter)
    

    tokenizer.print_tensor(data_tensor)
    print('\n---\n')
    
    dataset.print_batch()
    print('\n---\n')
    
    m.print_model(xb, yb)
    print('\n---\n')
    
    
    context = torch.zeros((1,1), dtype=torch.long, device=device)

    print(tokenizer.decode(m.generate(context, max_new_tokens=100)[0].tolist()))



if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python main.py <filepath>")
        exit(1)
        
    target_path = argv[1]
    main(target_path)
