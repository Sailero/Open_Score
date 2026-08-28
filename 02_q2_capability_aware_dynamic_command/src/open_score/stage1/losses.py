"""Minimal Double-Q one-step loss; a sequence replay trainer is the next step."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from open_score.contracts import GlobalState, TeamObservation
from open_score.stage1.entity_qmix import VariableScaleQMIX


@dataclass
class QMixTransition:
    observation: TeamObservation
    state: GlobalState
    actions: Tensor  # [batch, agents]
    reward: Tensor  # [batch]
    next_observation: TeamObservation
    next_state: GlobalState
    done: Tensor  # [batch], 1 for terminal transitions


def one_step_qmix_td_loss(
    online: VariableScaleQMIX,
    target: VariableScaleQMIX,
    batch: QMixTransition,
    gamma: float = 0.99,
    hidden: Optional[Tensor] = None,
    next_hidden: Optional[Tensor] = None,
) -> Tensor:
    """Double-Q target used by the stage-1 smoke test.

    Full training must replace this with burn-in plus sequence TD(lambda); this
    small function exists to make tensor semantics and gradients executable now.
    """

    q_values, _ = online.agent_q(batch.observation, hidden)
    chosen_q = q_values.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
    q_total = online.mix(chosen_q, batch.observation, batch.state)

    with torch.no_grad():
        next_online_q, _ = online.agent_q(batch.next_observation, next_hidden)
        next_actions = next_online_q.argmax(dim=-1)
        next_target_q, _ = target.agent_q(batch.next_observation, next_hidden)
        next_chosen_q = next_target_q.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
        next_total = target.mix(next_chosen_q, batch.next_observation, batch.next_state)
        td_target = batch.reward + gamma * (1.0 - batch.done.to(batch.reward.dtype)) * next_total

    return F.smooth_l1_loss(q_total, td_target)
