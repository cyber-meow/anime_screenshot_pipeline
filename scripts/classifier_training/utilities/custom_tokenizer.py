import torch
import pickle
from .build_vocab import Vocabulary

class CustomTokenizer(object):
    def __init__(self, vocab_path, max_text_seq_len, ret_tensor=True):
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
            self.vocab_size = len(self.vocab)
        self.max_text_seq_len = max_text_seq_len
        self.ret_tensor = ret_tensor

    def __call__(self, tag_list):
        no_tokens = len(tag_list) + 2
        diff = abs(self.max_text_seq_len - no_tokens)

        tokens = []
        tokens.append(self.vocab('[CLS]'))

        if no_tokens > self.max_text_seq_len:
            tokens.extend([self.vocab(tag) for tag in tag_list[:self.max_text_seq_len-2]])
            tokens.append(self.vocab('[SEP]'))
        elif no_tokens < self.max_text_seq_len:
            tokens.extend([self.vocab(tag) for tag in tag_list])
            tokens.append(self.vocab('[SEP]'))
            tokens.extend([self.vocab('[PAD]') for _ in range(diff)])
        else:
            tokens.extend([self.vocab(tag) for tag in tag_list])
            tokens.append(self.vocab('[SEP]'))

        if self.ret_tensor:
            return torch.tensor([tokens], dtype=torch.int64)
        return tokens
        
    def decode(self, tokens_list):
        if self.ret_tensor:
            tokens_notensor = tokens_list.squeeze().tolist()
            tag_list = [self.vocab.ret_word(idx) for idx in tokens_notensor]
            return tag_list
        else:
            return [self.vocab.ret_word(idx) for idx in tokens_list]

