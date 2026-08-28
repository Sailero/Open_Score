"""Two-timescale alternating update with a held-out scale guard and rollback."""

import copy
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AlternatingUpdateResult:
    accepted: bool
    old_worst_scale_success: float
    new_worst_scale_success: float
    loss: float


def stability_regularized_loss(
    td_loss: Tensor,
    current_q: Tensor,
    frozen_previous_q: Tensor,
    agent_mask: Tensor,
    stability_weight: float,
) -> Tensor:
    """Keep a QMIX executor from forgetting scales while the commander changes."""

    mask = agent_mask.to(current_q.dtype).unsqueeze(-1)
    distillation = ((current_q - frozen_previous_q.detach()).pow(2) * mask).sum()
    normalizer = mask.sum() * current_q.shape[-1]
    distillation = distillation / normalizer.clamp_min(1.0)
    return td_loss + stability_weight * distillation


def guarded_lower_update(
    lower_policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: Tensor,
    old_scale_success: Dict[str, float],
    evaluate_scale_success: Callable[[], Dict[str, float]],
    allowed_worst_drop: float = 0.01,
    max_grad_norm: float = 10.0,
) -> AlternatingUpdateResult:
    """Take one lower-level update and roll it back if worst-scale success drops.

    The outer loop is: freeze lower -> refresh S2/S3 -> freeze commander -> call
    this update on commander-induced assignments -> re-evaluate -> accept/rollback.
    """

    old_parameters = copy.deepcopy(lower_policy.state_dict())
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    old_worst = min(old_scale_success.values())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lower_policy.parameters(), max_grad_norm)
    optimizer.step()
    new_success = evaluate_scale_success()
    new_worst = min(new_success.values())
    accepted = new_worst >= old_worst - allowed_worst_drop
    if not accepted:
        lower_policy.load_state_dict(old_parameters)
        optimizer.load_state_dict(old_optimizer)
    return AlternatingUpdateResult(accepted, old_worst, new_worst, float(loss.detach().cpu()))
