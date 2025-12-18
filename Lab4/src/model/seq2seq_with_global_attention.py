"""
Seq2Seq model with Global Attention (Luong Attention).

This module implements the Luong attention mechanism where attention scores
are computed using dot-product or general multiplicative scoring functions.
"""

import torch
import torch.nn as nn

from src.data.vocab import Vocab
from .encoder import Encoder
from .decoder import Decoder


class GlobalAttention(nn.Module):
    """
    Global attention mechanism (Luong et al., 2015).
    
    Supports different scoring functions:
        - dot: score(h_t, h_s) = h_t^T * h_s
        - general: score(h_t, h_s) = h_t^T * W * h_s
    """
    
    def __init__(self, hidden_dim: int, attention_type: str = "general"):
        super().__init__()
        self.attention_type = attention_type
        
        # Weight matrix for 'general' scoring
        self.weight_matrix = (
            nn.Linear(hidden_dim, hidden_dim, bias=False) 
            if attention_type == "general" 
            else None
        )
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, hidden_dim)
            mask: (batch_size, src_len) - padding mask
            
        Returns:
            context: (batch_size, hidden_dim) - weighted sum of encoder outputs
            attention_weights: (batch_size, src_len) - attention distribution
        """
        # Apply weight transformation for 'general' attention
        query = decoder_hidden
        if self.weight_matrix is not None:
            query = self.weight_matrix(query)
        
        # Compute attention scores: (batch_size, src_len)
        scores = torch.bmm(
            encoder_outputs, 
            query.unsqueeze(2)
        ).squeeze(2)
        
        # Apply mask to ignore padding positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=1)
        
        # Compute context vector as weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1), 
            encoder_outputs
        ).squeeze(1)
        
        return context, attention_weights


class Seq2SeqWithGlobalAttention(nn.Module):
    """
    Sequence-to-Sequence model with global attention mechanism.
    
    Architecture:
        - Encoder: Bidirectional LSTM
        - Decoder: Unidirectional LSTM with global attention
        - Attention: Luong (global/multiplicative) attention mechanism
    """
    
    def __init__(
        self, 
        vocab: Vocab, 
        embed_dim: int, 
        hidden_dim: int, 
        n_layers: int, 
        dropout: float, 
        attention_type: str = "general"
    ):
        super().__init__()
        self.vocab = vocab
        
        # Core components
        self.encoder = Encoder(vocab, embed_dim, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(vocab, embed_dim, hidden_dim, n_layers, dropout)
        self.attention = GlobalAttention(hidden_dim, attention_type)
        
        # Attention combination layer
        self.combine_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab.tar_vocab_size)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)
    
    def forward(self, src_ids, tar_ids):
        """
        Forward pass with teacher forcing.
        
        Args:
            src_ids: (batch_size, src_len) - source token IDs
            tar_ids: (batch_size, tar_len) - target token IDs
            
        Returns:
            loss: scalar tensor
        """
        batch_size, tar_len = tar_ids.shape
        vocab_size = self.vocab.tar_vocab_size
        
        # Initialize outputs tensor
        outputs = torch.zeros(
            batch_size, tar_len, vocab_size, 
            device=src_ids.device
        )
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src_ids)
        
        # Create padding mask
        padding_mask = (src_ids != self.vocab.PAD_ID)
        
        # Decode with teacher forcing
        decoder_input = tar_ids[:, 0:1]
        for t in range(1, tar_len):
            # Decoder step (get raw LSTM output)
            decoder_output, hidden, cell = self.decoder(hidden, cell, decoder_input,
                                                       return_output=True)
            
            # Compute attention context
            context, _ = self.attention(
                hidden[-1], 
                encoder_outputs, 
                padding_mask
            )
            
            # Combine decoder hidden and context (Luong's approach)
            combined = torch.cat([hidden[-1], context], dim=1)
            attentional_hidden = torch.tanh(self.combine_layer(combined))
            
            # Project to vocabulary
            outputs[:, t] = self.output_projection(attentional_hidden)
            
            # Next input (teacher forcing)
            decoder_input = tar_ids[:, t:t+1]
        
        # Compute loss
        loss = self.criterion(
            outputs[:, 1:].reshape(-1, vocab_size),
            tar_ids[:, 1:].reshape(-1)
        )
        
        return loss
    
    @torch.no_grad()
    def predict(self, src_ids):
        """
        Generate translations using greedy decoding.
        
        Args:
            src_ids: (batch_size, src_len) - source token IDs
            
        Returns:
            predictions: (batch_size, generated_len) - generated token IDs
        """
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src_ids)
        padding_mask = (src_ids != self.vocab.PAD_ID)
        
        # Initialize with SOS token
        decoder_input = torch.full(
            (batch_size, 1), 
            self.vocab.SOS_ID, 
            dtype=torch.long, 
            device=device
        )
        
        # Generate tokens autoregressively
        predictions = []
        for _ in range(self.vocab.max_length):
            # Decoder step (get raw LSTM output)
            decoder_output, hidden, cell = self.decoder(hidden, cell, decoder_input,
                                                       return_output=True)
            
            # Attention
            context, _ = self.attention(
                hidden[-1], 
                encoder_outputs, 
                padding_mask
            )
            
            # Combine and project
            combined = torch.cat([hidden[-1], context], dim=1)
            attentional_hidden = torch.tanh(self.combine_layer(combined))
            logits = self.output_projection(attentional_hidden)
            
            # Greedy selection
            token = logits.argmax(dim=-1, keepdim=True)
            predictions.append(token)
            
            # Stop if all sequences generated EOS
            if (token == self.vocab.EOS_ID).all():
                break
                
            decoder_input = token
        
        return torch.cat(predictions, dim=1)
