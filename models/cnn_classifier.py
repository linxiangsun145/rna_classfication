from __future__ import annotations

import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int = 6,
        embedding_dim: int = 128,
        conv_channels: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(conv_channels, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        features = embedded.transpose(1, 2)
        features = self.activation(self.conv(features))

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).bool()
            features = features.masked_fill(~mask, float("-inf"))
            pooled = features.max(dim=-1).values
            invalid_rows = ~mask.any(dim=-1).squeeze(1)
            if invalid_rows.any():
                pooled[invalid_rows] = 0.0
        else:
            pooled = features.max(dim=-1).values

        pooled = self.dropout(pooled)
        return self.classifier(pooled)
