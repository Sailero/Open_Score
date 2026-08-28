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
