"""QMIX adapted to padded entity sets and a variable number of active agents."""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from open_score.contracts import GlobalState, TeamObservation
from open_score.nn import MaskedSetEncoder


class VariableEntityAgent(nn.Module):
    """One shared recurrent utility network for every agent and team size."""

    def __init__(
        self,
        entity_dim: int,
        self_dim: int,
        task_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.entity_encoder = MaskedSetEncoder(entity_dim, hidden_dim, hidden_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(hidden_dim + self_dim + task_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def initial_hidden(self, batch: int, agents: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, agents, self.hidden_dim, device=device)

    def forward(
        self, observation: TeamObservation, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        observation.validate()
        batch, agents, entities, entity_dim = observation.entity_obs.shape
        flat_entities = observation.entity_obs.reshape(batch * agents, entities, entity_dim)
        flat_mask = observation.entity_mask.reshape(batch * agents, entities)
        entity_context = self.entity_encoder(flat_entities, flat_mask).reshape(batch, agents, -1)
        inputs = self.input_layer(
            torch.cat([entity_context, observation.self_obs, observation.task_obs], dim=-1)
        )
        if hidden is None:
            hidden = self.initial_hidden(batch, agents, inputs.device)
        next_hidden = self.rnn(inputs.reshape(batch * agents, -1), hidden.reshape(batch * agents, -1))
        next_hidden = next_hidden.reshape(batch, agents, -1)
        q_values = self.q_head(next_hidden)
        q_values = q_values.masked_fill(~observation.avail_actions.bool(), -1e9)
        return q_values, next_hidden * observation.agent_mask.unsqueeze(-1).to(next_hidden.dtype)


class EntityMonotonicMixer(nn.Module):
    """A QMIX mixer whose dimensions do not depend on the number of agents.

    Positive state-conditioned weights retain dQ_tot/dQ_i >= 0.  Mean scaling
    avoids making the value magnitude grow mechanically with the roster size.
    """

    def __init__(
        self,
        state_entity_dim: int,
        agent_context_dim: int,
        hidden_dim: int = 64,
        mixing_dim: int = 32,
    ):
        super().__init__()
        self.state_encoder = MaskedSetEncoder(state_entity_dim, hidden_dim, hidden_dim)
        self.first_weights = nn.Linear(hidden_dim + agent_context_dim, mixing_dim)
        self.first_bias = nn.Linear(hidden_dim, mixing_dim)
        self.final_weights = nn.Linear(hidden_dim, mixing_dim)
        self.final_bias = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        chosen_agent_q: Tensor,
        state: GlobalState,
        agent_context: Tensor,
        agent_mask: Tensor,
    ) -> Tensor:
        state.validate()
        state_context = self.state_encoder(state.entities, state.entity_mask)
        agents = chosen_agent_q.shape[1]
        expanded_state = state_context.unsqueeze(1).expand(-1, agents, -1)
        positive_w1 = F.softplus(self.first_weights(torch.cat([expanded_state, agent_context], dim=-1)))
        active = agent_mask.to(chosen_agent_q.dtype).unsqueeze(-1)
        active_count = active.sum(dim=1).clamp_min(1.0)
        mixed_input = (
            positive_w1 * chosen_agent_q.unsqueeze(-1) * active
        ).sum(dim=1) / active_count
        hidden = F.elu(mixed_input + self.first_bias(state_context))
        positive_w2 = F.softplus(self.final_weights(state_context))
        return (positive_w2 * hidden).sum(dim=-1) + self.final_bias(state_context).squeeze(-1)


class VariableScaleQMIX(nn.Module):
    """Shared utility network plus the variable-cardinality monotonic mixer."""

    def __init__(
        self,
        entity_dim: int,
        self_dim: int,
        task_dim: int,
        state_entity_dim: int,
        action_dim: int,
        agent_hidden_dim: int = 128,
        mixer_hidden_dim: int = 64,
        mixing_dim: int = 32,
    ):
        super().__init__()
        self.agent = VariableEntityAgent(
            entity_dim, self_dim, task_dim, action_dim, hidden_dim=agent_hidden_dim
        )
        self.mixer = EntityMonotonicMixer(
            state_entity_dim,
            self_dim + task_dim,
            hidden_dim=mixer_hidden_dim,
            mixing_dim=mixing_dim,
        )

    def agent_q(
        self, observation: TeamObservation, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.agent(observation, hidden)

    def mix(self, chosen_q: Tensor, observation: TeamObservation, state: GlobalState) -> Tensor:
        context = torch.cat([observation.self_obs, observation.task_obs], dim=-1)
        return self.mixer(chosen_q, state, context, observation.agent_mask)

    @torch.no_grad()
    def act(
        self,
        observation: TeamObservation,
        hidden: Optional[Tensor] = None,
        epsilon: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        q_values, next_hidden = self.agent_q(observation, hidden)
        greedy = q_values.argmax(dim=-1)
        if epsilon <= 0.0:
            return greedy, next_hidden
        random_scores = torch.rand_like(q_values).masked_fill(~observation.avail_actions.bool(), -1.0)
        random_actions = random_scores.argmax(dim=-1)
        explore = torch.rand_like(greedy, dtype=torch.float32) < epsilon
        actions = torch.where(explore, random_actions, greedy)
        actions = torch.where(observation.agent_mask.bool(), actions, torch.zeros_like(actions))
        return actions, next_hidden
