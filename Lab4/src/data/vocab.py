import torch
import json
from typing import Union, List


class Vocab:
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = '<PAD>', '<SOS>', '<EOS>', '<UNK>'
    PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
    
    def __init__(self, data_file: str, max_length: int = 60, 
                 src_lang: str = 'vietnamese', tar_lang: str = 'english'):
        self.data_file = data_file
        self.max_length = max_length
        self.src_lang = src_lang
        self.tar_lang = tar_lang
        
        self._init_vocab_dicts()
        self._build_vocab()
        
        self.src_vocab_size = len(self.src_i2w)
        self.tar_vocab_size = len(self.tar_i2w)
    
    def _init_vocab_dicts(self):
        special_tokens = {
            self.PAD_TOKEN: self.PAD_ID,
            self.SOS_TOKEN: self.SOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.UNK_TOKEN: self.UNK_ID
        }
        self.src_w2i = dict(special_tokens)
        self.tar_w2i = dict(special_tokens)
        self.src_i2w = {}
        self.tar_i2w = {}

    
    def _build_vocab(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        src_words, tar_words = set(), set()
        for item in dataset:
            src_words.update(item[self.src_lang].lower().split())
            tar_words.update(item[self.tar_lang].lower().split())
        
        self._populate_mappings(src_words, self.src_w2i, self.src_i2w)
        self._populate_mappings(tar_words, self.tar_w2i, self.tar_i2w)
    
    def _populate_mappings(self, words: set, w2i: dict, i2w: dict):
        idx = len(w2i)
        for word in sorted(words):
            w2i[word] = idx
            idx += 1
        i2w.update({v: k for k, v in w2i.items()})
    
    def encode(self, text: str, lang: str) -> torch.Tensor:
        tokens = text.lower().split()
        w2i = self._get_w2i(lang)
        
        ids = [w2i.get(tok, self.UNK_ID) for tok in tokens]
        ids = ids[:self.max_length - 2]
        ids = [self.SOS_ID] + ids + [self.EOS_ID]
        ids += [self.PAD_ID] * (self.max_length - len(ids))
        
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids: Union[torch.Tensor, List], lang: str) -> str:
        i2w = self._get_i2w(lang)
        tokens = []
        
        for token_id in ids:
            idx = token_id.item() if hasattr(token_id, 'item') else int(token_id)
            if idx in {self.PAD_ID, self.SOS_ID}:
                continue
            if idx == self.EOS_ID:
                break
            tokens.append(i2w.get(idx, self.UNK_TOKEN))
        
        return ' '.join(tokens)
    
    def _get_w2i(self, lang: str) -> dict:
        if lang == self.src_lang:
            return self.src_w2i
        elif lang == self.tar_lang:
            return self.tar_w2i
        raise ValueError(f"Invalid language: {lang}")
    
    def _get_i2w(self, lang: str) -> dict:
        if lang == self.src_lang:
            return self.src_i2w
        elif lang == self.tar_lang:
            return self.tar_i2w
        raise ValueError(f"Invalid language: {lang}")
        

if __name__ == '__main__':
    import numpy as np
    
    vocab = Vocab(r'src\data\small-train.json', 100)
    print(f"Source vocab: {vocab.src_vocab_size}, Target vocab: {vocab.tar_vocab_size}")
    
    sample_en = "Hurricane Dorian made landfall as a Category 5 storm."
    encoded = vocab.encode(sample_en, 'english')
    decoded = vocab.decode(encoded, 'english')
    print(f"\nOriginal: {sample_en}")
    print(f"Decoded:  {decoded}")
