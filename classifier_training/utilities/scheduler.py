import math
import random
import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class MasksSchedule():

    def __init__(self, device, mask_schedule, masking_behavior, 
        tokenizer, vocab_size, batch_size, max_text_seq_len, 
        warmup_steps, cooldown_steps, total_steps, cycles=.5):
            
        self.device = device

        self.mask_schedule = mask_schedule
        self.masking_behavior = masking_behavior
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        
        self.batch_size = batch_size
        self.max_text_seq_len = max_text_seq_len
        
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.total_steps = total_steps
        self.cycles = cycles
    
        if self.tokenizer == 'wp':
            # 0 is [PAD], 101 is [CLS], 102 is [SEP]
            self.special_tokens = [0, 101, 102]
        elif self.tokenizer == 'tag':
            # 0 is [PAD], 2 is [CLS], 3 is [SEP]
            self.special_tokens = [0, 2, 3]

    def ret_mask(self, step, tokens_text=None):
        step = step[0]
        
        if self.mask_schedule == None:
            return None, None

        elif self.mask_schedule == 'bert':
            # 15 % masking like bert but only mask (0) or 1
            masks = torch.from_numpy(np.random.choice(a=[0, 1], size=(self.batch_size, self.max_text_seq_len), 
                p=[0.15, 0.85])).to(self.device)
        
        elif self.mask_schedule == 'full':
            # from beginning all masks equal to 1
            masks = torch.from_numpy(np.random.choice(a=[0, 1], size=(self.batch_size, self.max_text_seq_len), 
                p=[1, 0])).to(self.device)
            
        elif self.mask_schedule == 'sigmoid':
            # during warmup attend to all tokens
            # during cooldown attend to no text tokens
            # else attend to a percentage of text tokens following cosine function
            
            if step < self.warmup_steps:
                masks = torch.from_numpy(np.random.choice(a=[0, 1], size=(self.batch_size, self.max_text_seq_len), 
                    p=[0, 1])).to(self.device)
            
            elif step > (self.total_steps - self.cooldown_steps):
                masks = torch.from_numpy(np.random.choice(a=[0, 1], size=(self.batch_size, self.max_text_seq_len), 
                    p=[1, 0])).to(self.device)
            
            else:
                progress = (float(step - self.warmup_steps) / 
                    (float(max(1, self.total_steps - self.warmup_steps - self.cooldown_steps))))
                
                prob_visible = max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
                prob_mask = 1.0 - prob_visible
                
                masks = torch.from_numpy(np.random.choice(a=[0, 1], size=(self.batch_size, self.max_text_seq_len), 
                    p=[prob_mask, prob_visible])).to(self.device)
            
        if self.masking_behavior == 'constant':
            # if mask then change token to 1 (unused token)
            updated_numbers = torch.ones(self.batch_size, self.max_text_seq_len, dtype=torch.int64).to(self.device)
        elif self.masking_behavior == 'random':
            updated_numbers = torch.randint(0, self.vocab_size-1, (self.batch_size, self.max_text_seq_len)).to(self.device)
        
        tokens_text_updated = torch.where(
            (masks==0) & (tokens_text!=self.special_tokens[0]) & (tokens_text!=self.special_tokens[1]) & (tokens_text!=self.special_tokens[2]), 
            updated_numbers, tokens_text)
        labels_text = torch.where(
            (masks==1) | (tokens_text==self.special_tokens[0]) | (tokens_text==self.special_tokens[1]) | (tokens_text==self.special_tokens[2]), 
            -100, tokens_text)
        
        return tokens_text_updated, labels_text
