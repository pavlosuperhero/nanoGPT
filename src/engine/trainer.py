import torch
import logging
from torch import nn

class Trainer:
    def __init__(self, parameters, learning_rate: float):
        self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        self.logger = logging.getLogger(self.__class__.__name__)


    @torch.no_grad() 
    def estimate_loss(self, model, eval_iters: int, batch_generator):
        out = {}
        model.eval() 
        
        for split in ['train', 'val']: 
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = batch_generator.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item() 
            out[split] = losses.mean()
            
        model.train() 
        return out

    def count_loss(self, max_iters: int, eval_interval: int, batch_generator, model, eval_iters: int):
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = self.estimate_loss(model, eval_iters, batch_generator)
                self.logger.debug(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = batch_generator.get_batch('train')
            logits, loss = model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
