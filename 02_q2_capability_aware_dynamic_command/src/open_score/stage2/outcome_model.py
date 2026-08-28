"""Assignment-conditioned breach predictor for frozen stage-1 executors."""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from open_score.nn import MaskedSetAttentionEncoder


class BilateralOutcomeModel(nn.Module):
    """Predict P(any protected target is breached within H_o).

    Defender and attacker tokens contain an entity embedding concatenated with
    a task-assignment encoding.  Set pooling makes both rosters variable-sized.
    """

    def __init__(
        self,
        state_entity_dim: int,
        defender_token_dim: int,
        attacker_token_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_encoder = MaskedSetAttentionEncoder(
            state_entity_dim, hidden_dim, hidden_dim
        )
        self.defender_encoder = MaskedSetAttentionEncoder(
            defender_token_dim, hidden_dim, hidden_dim
        )
        self.attacker_encoder = MaskedSetAttentionEncoder(
            attacker_token_dim, hidden_dim, hidden_dim
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state_entities: Tensor,
        state_mask: Tensor,
        defender_tokens: Tensor,
        defender_mask: Tensor,
        attacker_tokens: Tensor,
        attacker_mask: Tensor,
    ) -> Tensor:
        state = self.state_encoder(state_entities, state_mask)
        defender = self.defender_encoder(defender_tokens, defender_mask)
        attacker = self.attacker_encoder(attacker_tokens, attacker_mask)
        return self.head(torch.cat([state, defender, attacker], dim=-1)).squeeze(-1)

    def breach_probability(self, *args: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(*args))


def selection_focused_loss(
    logits: Tensor,
    breach_labels: Tensor,
    selection_weights: Tensor,
    ranking_pairs: Optional[Tuple[Tensor, Tensor]] = None,
    ranking_weight: float = 0.2,
) -> Tensor:
    """Weighted probability loss plus within-snapshot pairwise ranking.

    `selection_weights` are frozen from a pilot selector; they must not be
    recomputed from the same labels used by this loss.
    """

    point_loss = F.binary_cross_entropy_with_logits(logits, breach_labels.float(), reduction="none")
    weights = selection_weights.float().clamp_min(0.0)
    point_loss = (point_loss * weights).sum() / weights.sum().clamp_min(1.0)
    if ranking_pairs is None:
        return point_loss
    safer_index, riskier_index = ranking_pairs
    # Lower logit means safer; enforce riskier_logit > safer_logit.
    rank_loss = F.softplus(logits[safer_index] - logits[riskier_index]).mean()
    return point_loss + ranking_weight * rank_loss
