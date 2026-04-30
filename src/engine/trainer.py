import torch
import logging
from torch import device, nn

class Trainer:
    def __init__(self, parameters, learning_rate: float, device: str):
        self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device


    @torch.no_grad() 
    def estimate_loss(self, model, eval_iters: int, batch_generator):
        out = {}
        model.eval() 
        
        for split in ['train', 'val']: 
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = batch_generator.get_batch(split, device=self.device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(X, Y)
                losses[k] = loss.item() 
            out[split] = losses.mean()
            
        model.train() 
        return out
    def count_loss(self, max_iters: int, eval_interval: int, batch_generator, model, eval_iters: int):
        gradient_accumulation_steps = 11
        
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = self.estimate_loss(model, eval_iters, batch_generator)
                self.logger.debug(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            self.optimizer.zero_grad(set_to_none=True)
            
            for micro_step in range(gradient_accumulation_steps):
                xb, yb = batch_generator.get_batch('train', device=self.device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(xb, yb)
                    loss = loss / gradient_accumulation_steps
                
                loss.backward()

            self.optimizer.step()
