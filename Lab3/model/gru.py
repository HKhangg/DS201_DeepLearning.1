import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from data.vocab import Vocab

class GRU(nn.Module):
    def __init__(self, vocabulary: Vocab, embedding_size = 256, hidden_size =256, layer_count = 5):
        super().__init__()
        self.vocabulary = vocabulary

        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary.vocab_size,
            embedding_dim=embedding_size,
            padding_idx=vocabulary.padding_idx
        )
        self.rnn_network = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=layer_count,
            batch_first=True
        )
        self.output_layer = nn.Linear(
            in_features=hidden_size,
            out_features=vocabulary.num_labels
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, token_ids: torch.Tensor, targets=None):
        seq_lengths = (token_ids != self.vocabulary.padding_idx).sum(dim=1)
        embedded_tokens = self.token_embedding(token_ids)

        packed_input = pack_padded_sequence(
            input=embedded_tokens,
            lengths=seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, final_hidden = self.rnn_network(packed_input)

        final_state = final_hidden[-1]

        prediction_logits = self.output_layer(final_state)

        if targets is not None:
            computed_loss = self.criterion(prediction_logits, targets.view(-1))
            return computed_loss, prediction_logits
        
        return prediction_logits

    