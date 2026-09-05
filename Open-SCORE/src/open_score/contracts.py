"""Tensor contracts retained for the frozen local executor."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class TeamObservation:
    """Padded, masked observations for one cooperative team.

    Shapes:
        entity_obs: [batch, agents, entities, entity_features]
        entity_mask: [batch, agents, entities], True for a real visible entity
        self_obs: [batch, agents, self_features]
        task_obs: [batch, agents, task_features]
        agent_mask: [batch, agents], True for an active controlled agent
        avail_actions: [batch, agents, actions], True for an available action
        action_entity_index: optional [batch, agents, actions] mapping a
            target-selecting action to its entity row, or -1 for a fixed
            non-target action
        action_target_type: optional [batch, agents, actions] target kind
            (0 non-target, 1 enemy/damage, 2 ally/heal, 3 protected asset)
    """

    entity_obs: Tensor
    entity_mask: Tensor
    self_obs: Tensor
    task_obs: Tensor
    agent_mask: Tensor
    avail_actions: Tensor
    action_entity_index: Optional[Tensor] = None
    action_target_type: Optional[Tensor] = None

    def validate(self) -> None:
        batch, agents, entities, _ = self.entity_obs.shape
        if self.entity_mask.shape != (batch, agents, entities):
            raise ValueError("entity_mask does not match entity_obs")
        if self.self_obs.shape[:2] != (batch, agents):
            raise ValueError("self_obs does not match agent axes")
        if self.task_obs.shape[:2] != (batch, agents):
            raise ValueError("task_obs does not match agent axes")
        if self.agent_mask.shape != (batch, agents):
            raise ValueError("agent_mask does not match agent axes")
        if self.avail_actions.shape[:2] != (batch, agents):
            raise ValueError("avail_actions does not match agent axes")
        action_shape = self.avail_actions.shape
        if self.action_entity_index is not None:
            if self.action_entity_index.shape != action_shape:
                raise ValueError("action_entity_index does not match avail_actions")
            indices = self.action_entity_index
            if torch.any(indices < -1) or torch.any(indices >= entities):
                raise ValueError("action_entity_index contains an invalid entity row")
            if torch.any(indices >= 0) and self.action_target_type is None:
                raise ValueError("target actions require explicit action_target_type")
        if self.action_target_type is not None:
            if self.action_target_type.shape != action_shape:
                raise ValueError("action_target_type does not match avail_actions")
            if torch.any(self.action_target_type < 0) or torch.any(
                self.action_target_type > 3
            ):
                raise ValueError("action_target_type must use the registered 0..3 enum")
            if self.action_entity_index is None:
                raise ValueError("action_target_type requires action_entity_index")
            target = self.action_entity_index >= 0
            typed = self.action_target_type > 0
            if torch.any(target != typed):
                raise ValueError("target action indices and target types disagree")
        visible = self.entity_mask.any(dim=-1)
        if torch.any(self.agent_mask.bool() & ~visible):
            raise ValueError("each active agent must observe at least one entity")

    def to(self, device: torch.device) -> "TeamObservation":
        return TeamObservation(
            **{
                name: value.to(device) if value is not None else None
                for name, value in self.__dict__.items()
            }
        )


@dataclass
class GlobalState:
    """Permutation-invariant global state used only during centralised training."""

    entities: Tensor  # [batch, entities, state_features]
    entity_mask: Tensor  # [batch, entities]

    def validate(self) -> None:
        if self.entities.shape[:2] != self.entity_mask.shape:
            raise ValueError("global entity_mask does not match entities")
        if torch.any(~self.entity_mask.bool().any(dim=-1)):
            raise ValueError("each state must contain at least one real entity")

    def to(self, device: torch.device) -> "GlobalState":
        return GlobalState(self.entities.to(device), self.entity_mask.to(device))
