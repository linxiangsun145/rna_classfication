from __future__ import annotations

import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int = 6,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(output_dim, num_classes)

    def masked_mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked_hidden.sum(dim=1) / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        hidden_states, _ = self.gru(embedded)
        pooled = self.masked_mean_pool(hidden_states, attention_mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
