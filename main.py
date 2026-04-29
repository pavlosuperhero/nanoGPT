from sys import argv
from sys import exit
from src.tokenizer.main import Tokenizer
from src.training.main import TrainingBatch

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
    training = TrainingBatch(data=data_tensor, batch_size=batch_size, block_size=block_size)
    data_batch = training.get_batch('train')

    tokenizer.print_tensor(data_tensor)
    print('\n---\n')
    training.print_batch()

if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python main.py <filepath>")
        exit(1)
        
    target_path = argv[1]
    main(target_path)
