from __future__ import annotations

import torch
from torch import nn


try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class MambaClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int = 6,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        classifier_head: str = "linear",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba-ssm is not installed. Install it before training with "
                "--model mamba, for example: pip install mamba-ssm"
            )

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(d_model),
                        "mamba": Mamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier_head = classifier_head
        if classifier_head == "linear":
            self.classifier = nn.Linear(d_model, num_classes)
        elif classifier_head == "mlp":
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unsupported classifier_head: {classifier_head}")

    def masked_mean_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked_hidden.sum(dim=1) / denom

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)

        for layer in self.layers:
            residual = hidden_states
            hidden_states = layer["norm"](hidden_states)
            hidden_states = layer["mamba"](hidden_states)
            hidden_states = hidden_states + residual

        hidden_states = self.final_norm(hidden_states)
        pooled = self.masked_mean_pool(hidden_states, attention_mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
