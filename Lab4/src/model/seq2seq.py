import torch
import torch.nn as nn

from src.data.vocab import Vocab
from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self, vocab: Vocab, embed_dim: int, hidden_dim: int, 
                 n_layers: int, dropout: float):
        super().__init__()
        self.vocab = vocab
        
        self.encoder = Encoder(vocab, embed_dim, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(vocab, embed_dim, hidden_dim, n_layers, dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)
    
    def forward(self, src_ids, tar_ids):
        batch_size, tar_len = tar_ids.shape
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, tar_len, vocab_size, device=src_ids.device)
        _, h, c = self.encoder(src_ids)
        
        decoder_input = tar_ids[:, 0:1]
        for t in range(1, tar_len):
            logits, h, c = self.decoder(h, c, decoder_input)
            outputs[:, t] = logits.squeeze(1)
            decoder_input = tar_ids[:, t:t+1]
        
        loss = self.criterion(outputs[:, 1:].reshape(-1, vocab_size),
                             tar_ids[:, 1:].reshape(-1))
        return loss
    
    @torch.no_grad()
    def predict(self, src_ids):
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        
        _, h, c = self.encoder(src_ids)
        decoder_input = torch.full((batch_size, 1), self.vocab.SOS_ID, 
                                   dtype=torch.long, device=device)
        
        predictions = []
        for _ in range(self.vocab.max_length):
            logits, h, c = self.decoder(h, c, decoder_input)
            token = logits.argmax(dim=-1)
            predictions.append(token)
            
            if (token == self.vocab.EOS_ID).all():
                break
            decoder_input = token
        
        return torch.cat(predictions, dim=1)


        return predictions