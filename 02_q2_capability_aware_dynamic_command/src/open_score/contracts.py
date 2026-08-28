"""Tensor contracts shared by environments and all four research stages."""

from dataclasses import dataclass

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
    """

    entity_obs: Tensor
    entity_mask: Tensor
    self_obs: Tensor
    task_obs: Tensor
    agent_mask: Tensor
    avail_actions: Tensor

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
        visible = self.entity_mask.any(dim=-1)
        if torch.any(self.agent_mask.bool() & ~visible):
            raise ValueError("each active agent must observe at least one entity")

    def to(self, device: torch.device) -> "TeamObservation":
        return TeamObservation(*[value.to(device) for value in self.__dict__.values()])


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
