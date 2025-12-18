import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.vocab import Vocab


class Encoder(nn.Module):
    def __init__(self, vocab: Vocab, embed_dim: int = 256, hidden_dim: int = 256,
                 n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.vocab = vocab
        
        self.embed = nn.Embedding(vocab.src_vocab_size, embed_dim, 
                                   padding_idx=vocab.PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_ids: torch.Tensor):
        lengths = (src_ids != self.vocab.PAD_ID).sum(dim=1).cpu()
        
        embedded = self.dropout(self.embed(src_ids))
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, 
                                     enforce_sorted=False)
        
        packed_out, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, 
                                         total_length=src_ids.size(1))
        
        return outputs, h, c
