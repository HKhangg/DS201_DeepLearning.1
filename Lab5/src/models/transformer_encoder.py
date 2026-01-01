import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional Encoding theo paper 'Attention is All You Need'"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Tạo positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention theo paper 'Attention is All You Need'"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, num_heads, seq_len, d_k]
            K: [batch_size, num_heads, seq_len, d_k]
            V: [batch_size, num_heads, seq_len, d_k]
            mask: [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] - padding mask
        """
        batch_size = query.size(0)
        
        # Linear projections và reshape cho multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Reshape mask nếu có
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        
        # Scaled dot-product attention
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(x)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Một lớp Transformer Encoder"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len]
        """
        # Self-attention với residual connection và layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward với residual connection và layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder với N lớp"""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=3, 
                 d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack của N encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len] - input token ids
            mask: [batch_size, seq_len] - padding mask (1 for valid, 0 for padding)
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Embedding và scale theo paper
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.positional_encoding(x)
        
        # Qua từng encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        return x


class TransformerForSequenceClassification(nn.Module):
    """Transformer Encoder cho bài toán phân loại sequence (Bài 1)"""
    
    def __init__(self, vocab_size, num_classes, d_model=512, num_heads=8, 
                 num_layers=3, d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super(TransformerForSequenceClassification, self).__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # Encoder output
        encoder_output = self.encoder(input_ids, attention_mask)
        
        # Pooling: lấy [CLS] token (position 0) hoặc mean pooling
        # Ở đây dùng mean pooling trên các token không phải padding
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_output.size()).float()
            sum_embeddings = torch.sum(encoder_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(encoder_output, dim=1)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class TransformerForTokenClassification(nn.Module):
    """Transformer Encoder cho bài toán gán nhãn token (Bài 2)"""
    
    def __init__(self, vocab_size, num_labels, d_model=512, num_heads=8, 
                 num_layers=3, d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super(TransformerForTokenClassification, self).__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # Token classification head
        self.classifier = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, num_labels]
        """
        # Encoder output
        encoder_output = self.encoder(input_ids, attention_mask)
        encoder_output = self.dropout(encoder_output)
        
        # Classify mỗi token
        logits = self.classifier(encoder_output)
        
        return logits
