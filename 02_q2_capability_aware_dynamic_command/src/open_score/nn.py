"""Small permutation-invariant neural building blocks."""

import torch
from torch import Tensor, nn


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    total = (values * weights).sum(dim=dim)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return total / count


class MaskedSetEncoder(nn.Module):
    """DeepSets-style encoder that is invariant to entity order and padding."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, output_dim),
            nn.ReLU(),
        )

    def forward(self, entities: Tensor, mask: Tensor) -> Tensor:
        encoded = self.element(entities)
        mean = masked_mean(encoded, mask, dim=-2)
        neg_inf = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.bool().unsqueeze(-1), neg_inf).max(dim=-2).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        count = torch.log1p(mask.to(encoded.dtype).sum(dim=-1, keepdim=True))
        return self.output(torch.cat([mean, maximum, count], dim=-1))


class MaskedSetAttentionEncoder(nn.Module):
    """Small Set-Transformer-style encoder for bilateral outcome tokens.

    It uses masked self-attention followed by attention pooling from a learned
    seed.  The parameters are independent of set cardinality and the output is
    invariant to a common permutation of tokens and masks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.element = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.pool_seed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + 1, output_dim),
            nn.ReLU(),
        )

    def forward(self, entities: Tensor, mask: Tensor) -> Tensor:
        if torch.any(~mask.bool().any(dim=-1)):
            raise ValueError("attention set encoder requires one active token per set")
        encoded = self.element(entities)
        padding_mask = ~mask.bool()
        attended, _ = self.self_attention(
            encoded, encoded, encoded, key_padding_mask=padding_mask, need_weights=False
        )
        encoded = self.self_norm(encoded + attended)
        seed = self.pool_seed.expand(encoded.shape[0], -1, -1)
        pooled, _ = self.pool_attention(
            seed, encoded, encoded, key_padding_mask=padding_mask, need_weights=False
        )
        count = torch.log1p(mask.to(encoded.dtype).sum(dim=-1, keepdim=True))
        return self.output(torch.cat([pooled.squeeze(1), count], dim=-1))
