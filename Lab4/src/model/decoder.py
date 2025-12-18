import torch
import torch.nn as nn

from src.data.vocab import Vocab


class Decoder(nn.Module):
    def __init__(self, vocab: Vocab, embed_dim: int = 256, hidden_dim: int = 256,
                 n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.vocab = vocab
        
        self.embed = nn.Embedding(vocab.tar_vocab_size, embed_dim, 
                                  padding_idx=vocab.PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab.tar_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, c: torch.Tensor, tar_ids: torch.Tensor, 
                return_output=False):
        embedded = self.dropout(self.embed(tar_ids))
        output, (h, c) = self.lstm(embedded, (h, c))
        
        if return_output:
            return output, h, c
        
        logits = self.fc(output)
        return logits, h, c
