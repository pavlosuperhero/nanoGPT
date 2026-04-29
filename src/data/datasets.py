import torch
import logging

class BatchGenerator:
    def __init__(
            self,
            data: torch.Tensor,
            batch_size: int = 4,
            block_size: int = 8,
            manual_seed: int = 1337,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        torch.manual_seed(manual_seed)
        
        self.batch_size = batch_size
        self.block_size = block_size

        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, device: str = 'cpu'):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size, ))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        x, y = x.to(device), y.to(device) 
        return x, y

    def print_batch(self):
        xb, yb = self.get_batch('train')
        
        self.logger.debug(f"Inputs shape: {xb.shape}\n{xb}\n---")
        self.logger.debug(f"Targets shape: {yb.shape}\n{yb}\n---")

        for b in range(self.batch_size):
            for t in range(self.block_size):
                context = xb[b, :t+1]
                target = yb[b, t]
                self.logger.debug(f"When inputs is {context.tolist()} the target: {target.item()}")
