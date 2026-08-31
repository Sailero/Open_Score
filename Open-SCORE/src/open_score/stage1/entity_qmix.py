"""QMIX adapted to padded entity sets and a variable number of active agents."""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from open_score.contracts import GlobalState, TeamObservation
from open_score.nn import MaskedSetEncoder


class SingleAgentQueryAttentionEncoder(nn.Module):
    """Original Open-SCORE single-agent-query entity cross-attention.

    This is inspired by the design ideas of REFIL Attention-QMIX and SPECTra
    SAQA, but is an independent PyTorch implementation rather than a
    reproduction of either codebase.
    """

    def __init__(
        self,
        entity_dim: int,
        query_dim: int,
        hidden_dim: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        if attention_heads < 1 or hidden_dim % attention_heads:
            raise ValueError("attention_heads must divide hidden_dim")
        self.attention_heads = int(attention_heads)
        self.entity_embedding = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.Tanh(),
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, self.attention_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        entities: Tensor,
        mask: Tensor,
        query_features: Tensor,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        entity_embeddings = self.entity_embedding(entities)
        valid = mask.bool()
        if visibility_mask is not None:
            if visibility_mask.shape != mask.shape:
                raise ValueError("visibility_mask must match the entity mask")
            valid = valid & visibility_mask.bool()
        has_entity = valid.any(dim=-1)
        # MultiheadAttention rejects an all-padding row. Padding-only agent
        # slots receive a zero sentinel and are zeroed again after attention.
        safe_valid = valid.clone()
        if bool((~has_entity).any()):
            safe_valid[~has_entity, 0] = True
            entity_embeddings = entity_embeddings.clone()
            entity_embeddings[~has_entity, 0] = 0.0
        query = self.query_projection(query_features).unsqueeze(1)
        attended, _ = self.cross_attention(
            query,
            entity_embeddings,
            entity_embeddings,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        context = self.output(self.norm(query + attended)).squeeze(1)
        context = torch.where(
            has_entity.unsqueeze(-1), context, torch.zeros_like(context)
        )
        return context, entity_embeddings


class PermutationEquivariantActionHead(nn.Module):
    """Fixed-action head plus a shared scorer for entity-target actions.

    ``action_entity_index`` is the sole authority for the environment's action
    slot semantics.  The same scorer is applied to every selected entity, so a
    permutation of entity rows (and the corresponding index mapping) cannot
    change the represented policy.  Target type is explicit because SMAClite
    uses the same numeric slots for enemy damage and ally healing actions.
    """

    TARGET_TYPE_COUNT = 4

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        type_dim: int = 8,
        entity_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        entity_embedding_dim = entity_embedding_dim or hidden_dim
        self.fixed_head = nn.Linear(hidden_dim, action_dim)
        self.target_type_embedding = nn.Embedding(self.TARGET_TYPE_COUNT, type_dim)
        self.target_scorer = nn.Sequential(
            nn.Linear(hidden_dim + entity_embedding_dim + type_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        latent: Tensor,
        entity_embeddings: Tensor,
        action_entity_index: Optional[Tensor],
        action_target_type: Optional[Tensor],
    ) -> Tensor:
        scores = self.fixed_head(latent)
        if action_entity_index is None:
            return scores
        if action_target_type is None:
            raise ValueError("target action mapping requires explicit target types")
        target_mask = action_entity_index >= 0
        if not bool(target_mask.any()):
            return scores
        safe_index = action_entity_index.clamp_min(0)
        expanded_entities = entity_embeddings.unsqueeze(2).expand(
            -1, -1, scores.shape[-1], -1, -1
        )
        gather_index = safe_index.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, 1, entity_embeddings.shape[-1]
        )
        selected_entities = torch.gather(
            expanded_entities, dim=3, index=gather_index
        ).squeeze(3)
        expanded_latent = latent.unsqueeze(2).expand(-1, -1, scores.shape[-1], -1)
        target_types = self.target_type_embedding(action_target_type.clamp(0, 3))
        target_scores = self.target_scorer(
            torch.cat([expanded_latent, selected_entities, target_types], dim=-1)
        ).squeeze(-1)
        return torch.where(target_mask, target_scores, scores)


class VariableEntityAgent(nn.Module):
    """One shared recurrent utility network for every agent and team size."""

    def __init__(
        self,
        entity_dim: int,
        self_dim: int,
        task_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        encoder_kind: str = "deepset",
        attention_heads: int = 4,
        attention_embed_dim: Optional[int] = None,
        include_last_action: bool = False,
    ):
        super().__init__()
        if encoder_kind not in {"deepset", "saqa", "refil"}:
            raise ValueError("encoder_kind must be 'deepset', 'saqa' or 'refil'")
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.encoder_kind = encoder_kind
        self.attention_heads = int(attention_heads)
        self.include_last_action = bool(include_last_action)
        context_dim = hidden_dim
        if encoder_kind == "deepset":
            self.entity_encoder: nn.Module = MaskedSetEncoder(
                entity_dim, hidden_dim, hidden_dim
            )
        else:
            context_dim = int(attention_embed_dim or hidden_dim)
            self.entity_encoder = SingleAgentQueryAttentionEncoder(
                entity_dim,
                self_dim + task_dim,
                context_dim,
                self.attention_heads,
            )
        self.input_layer = nn.Sequential(
            nn.Linear(
                context_dim
                + self_dim
                + task_dim
                + (action_dim if self.include_last_action else 0),
                hidden_dim,
            ),
            nn.ReLU(),
        )
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.q_head = PermutationEquivariantActionHead(
            hidden_dim, action_dim, entity_embedding_dim=context_dim
        )

    def initial_hidden(self, batch: int, agents: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, agents, self.hidden_dim, device=device)

    def forward(
        self,
        observation: TeamObservation,
        hidden: Optional[Tensor] = None,
        last_action: Optional[Tensor] = None,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        observation.validate()
        batch, agents, entities, entity_dim = observation.entity_obs.shape
        flat_entities = observation.entity_obs.reshape(batch * agents, entities, entity_dim)
        flat_mask = observation.entity_mask.reshape(batch * agents, entities)
        if self.encoder_kind == "deepset":
            entity_embeddings = self.entity_encoder.element(flat_entities).reshape(
                batch, agents, entities, -1
            )
            entity_context = self.entity_encoder(flat_entities, flat_mask).reshape(
                batch, agents, -1
            )
        else:
            flat_query = torch.cat(
                [observation.self_obs, observation.task_obs], dim=-1
            ).reshape(batch * agents, -1)
            flat_visibility = (
                None
                if visibility_mask is None
                else visibility_mask.reshape(batch * agents, entities)
            )
            flat_context, flat_embeddings = self.entity_encoder(
                flat_entities, flat_mask, flat_query, flat_visibility
            )
            entity_context = flat_context.reshape(batch, agents, -1)
            entity_embeddings = flat_embeddings.reshape(
                batch, agents, entities, -1
            )
        features = [entity_context, observation.self_obs, observation.task_obs]
        if self.include_last_action:
            if last_action is None:
                last_action = torch.zeros(
                    batch,
                    agents,
                    self.action_dim,
                    dtype=observation.self_obs.dtype,
                    device=observation.self_obs.device,
                )
            if last_action.shape != (batch, agents, self.action_dim):
                raise ValueError("last_action has an incompatible shape")
            features.append(last_action.to(observation.self_obs.dtype))
        inputs = self.input_layer(torch.cat(features, dim=-1))
        if hidden is None:
            hidden = self.initial_hidden(batch, agents, inputs.device)
        next_hidden = self.rnn(inputs.reshape(batch * agents, -1), hidden.reshape(batch * agents, -1))
        next_hidden = next_hidden.reshape(batch, agents, -1)
        q_values = self.q_head(
            next_hidden,
            entity_embeddings,
            observation.action_entity_index,
            observation.action_target_type,
        )
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


class EntityAttentionFlexMixer(nn.Module):
    """REFIL-style variable-cardinality monotonic mixing network.

    Agent-conditioned cross-attention creates one positive row of first-layer
    mixing weights per active agent.  A learned global query creates the bias,
    final weights and state value.  The optional per-agent visibility masks are
    used by REFIL's imagined within-group/cross-group auxiliary objective.
    """

    def __init__(
        self,
        state_entity_dim: int,
        agent_context_dim: int,
        hypernet_hidden_dim: int = 128,
        mixing_dim: int = 32,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if hypernet_hidden_dim % attention_heads:
            raise ValueError("attention_heads must divide hypernet_hidden_dim")
        self.attention_heads = int(attention_heads)
        self.mixing_dim = int(mixing_dim)
        self.entity_embedding = nn.Sequential(
            nn.Linear(state_entity_dim, hypernet_hidden_dim),
            nn.ReLU(),
        )
        self.agent_query = nn.Sequential(
            nn.Linear(agent_context_dim, hypernet_hidden_dim),
            nn.Tanh(),
        )
        self.agent_attention = nn.MultiheadAttention(
            hypernet_hidden_dim, self.attention_heads, batch_first=True
        )
        self.global_query = nn.Parameter(torch.zeros(1, 1, hypernet_hidden_dim))
        nn.init.normal_(self.global_query, std=0.02)
        self.global_attention = nn.MultiheadAttention(
            hypernet_hidden_dim, self.attention_heads, batch_first=True
        )
        self.first_weight = nn.Linear(hypernet_hidden_dim, mixing_dim)
        self.first_bias = nn.Linear(hypernet_hidden_dim, mixing_dim)
        self.final_weight = nn.Linear(hypernet_hidden_dim, mixing_dim)
        self.state_value = nn.Linear(hypernet_hidden_dim, 1)

    def _global_context(self, embedded: Tensor, state_mask: Tensor) -> Tensor:
        batch = embedded.shape[0]
        query = self.global_query.expand(batch, -1, -1)
        context, _ = self.global_attention(
            query,
            embedded,
            embedded,
            key_padding_mask=~state_mask.bool(),
            need_weights=False,
        )
        return context.squeeze(1)

    def _agent_context(
        self,
        embedded: Tensor,
        state_mask: Tensor,
        agent_context: Tensor,
        visibility_mask: Optional[Tensor],
    ) -> Tensor:
        query = self.agent_query(agent_context)
        attention_mask = None
        if visibility_mask is not None:
            if visibility_mask.shape != (
                embedded.shape[0],
                agent_context.shape[1],
                embedded.shape[1],
            ):
                raise ValueError("mixer visibility mask has an incompatible shape")
            valid = visibility_mask.bool() & state_mask.bool().unsqueeze(1)
            missing = ~valid.any(dim=-1)
            if bool(missing.any()):
                valid = valid.clone()
                first_valid = state_mask.to(torch.int64).argmax(dim=-1)
                rows = missing.nonzero(as_tuple=False)
                valid[rows[:, 0], rows[:, 1], first_valid[rows[:, 0]]] = True
            attention_mask = (~valid).repeat_interleave(
                self.attention_heads, dim=0
            )
        context, _ = self.agent_attention(
            query,
            embedded,
            embedded,
            attn_mask=attention_mask,
            key_padding_mask=~state_mask.bool(),
            need_weights=False,
        )
        return context

    def _weights(
        self,
        state: GlobalState,
        agent_context: Tensor,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        state.validate()
        embedded = self.entity_embedding(state.entities)
        agent_latent = self._agent_context(
            embedded, state.entity_mask, agent_context, visibility_mask
        )
        global_context = self._global_context(embedded, state.entity_mask)
        positive_first = torch.softmax(self.first_weight(agent_latent), dim=-1)
        return positive_first, global_context

    def _finish(
        self,
        contributions: Tensor,
        global_context: Tensor,
    ) -> Tensor:
        hidden = F.elu(contributions + self.first_bias(global_context))
        positive_final = torch.softmax(
            self.final_weight(global_context), dim=-1
        )
        return (
            (positive_final * hidden).sum(dim=-1)
            + self.state_value(global_context).squeeze(-1)
        )

    def forward(
        self,
        chosen_agent_q: Tensor,
        state: GlobalState,
        agent_context: Tensor,
        agent_mask: Tensor,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tensor:
        first, global_context = self._weights(
            state, agent_context, visibility_mask
        )
        active = agent_mask.to(chosen_agent_q.dtype).unsqueeze(-1)
        contributions = (
            first * chosen_agent_q.unsqueeze(-1) * active
        ).sum(dim=1)
        return self._finish(contributions, global_context)

    def imagined(
        self,
        within_q: Tensor,
        interaction_q: Tensor,
        state: GlobalState,
        agent_context: Tensor,
        agent_mask: Tensor,
        within_visibility: Tensor,
        interaction_visibility: Tensor,
    ) -> Tensor:
        within_weights, global_context = self._weights(
            state, agent_context, within_visibility
        )
        interaction_weights, _ = self._weights(
            state, agent_context, interaction_visibility
        )
        active = agent_mask.to(within_q.dtype).unsqueeze(-1)
        contributions = (
            within_weights * within_q.unsqueeze(-1) * active
            + interaction_weights * interaction_q.unsqueeze(-1) * active
        ).sum(dim=1)
        return self._finish(contributions, global_context)


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
        encoder_kind: str = "deepset",
        attention_heads: int = 4,
        attention_embed_dim: Optional[int] = None,
        hypernet_hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder_kind = encoder_kind
        self.attention_heads = int(attention_heads)
        self.is_refil = encoder_kind == "refil"
        self.architecture_name = (
            "REFIL-QMIX-HAD-v1"
            if self.is_refil
            else (
                "OpenSCORE-SAQA-QMIX-SP"
                if encoder_kind == "saqa"
                else "OpenSCORE-DeepSet-QMIX"
            )
        )
        self.agent = VariableEntityAgent(
            entity_dim,
            self_dim,
            task_dim,
            action_dim,
            hidden_dim=agent_hidden_dim,
            encoder_kind=encoder_kind,
            attention_heads=attention_heads,
            attention_embed_dim=attention_embed_dim,
            include_last_action=self.is_refil,
        )
        if self.is_refil:
            self.mixer: nn.Module = EntityAttentionFlexMixer(
                state_entity_dim,
                self_dim + task_dim,
                hypernet_hidden_dim=hypernet_hidden_dim,
                mixing_dim=mixing_dim,
                attention_heads=attention_heads,
            )
        else:
            self.mixer = EntityMonotonicMixer(
                state_entity_dim,
                self_dim + task_dim,
                hidden_dim=mixer_hidden_dim,
                mixing_dim=mixing_dim,
            )

    def agent_q(
        self,
        observation: TeamObservation,
        hidden: Optional[Tensor] = None,
        last_action: Optional[Tensor] = None,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.agent(
            observation,
            hidden,
            last_action=last_action,
            visibility_mask=visibility_mask,
        )

    def mix(self, chosen_q: Tensor, observation: TeamObservation, state: GlobalState) -> Tensor:
        context = torch.cat([observation.self_obs, observation.task_obs], dim=-1)
        return self.mixer(chosen_q, state, context, observation.agent_mask)

    @staticmethod
    def imagination_visibility(
        observation: TeamObservation,
        group_a: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return same-group and cross-group masks for one episode split."""

        batch, agents, entities = observation.entity_mask.shape
        if group_a.shape != (batch, entities):
            raise ValueError("group assignment must have shape [batch, entities]")
        # HAD includes an explicit is-self feature in the final entity column.
        # This remains reliable under entity permutations and padded batching.
        self_rows = observation.entity_obs[..., -1].argmax(dim=-1)
        self_group = torch.gather(group_a, 1, self_rows)
        same = group_a.unsqueeze(1) == self_group.unsqueeze(-1)
        return same, ~same

    def mix_imagined(
        self,
        within_q: Tensor,
        interaction_q: Tensor,
        observation: TeamObservation,
        state: GlobalState,
        group_a: Tensor,
    ) -> Tensor:
        if not self.is_refil or not isinstance(
            self.mixer, EntityAttentionFlexMixer
        ):
            raise RuntimeError("imagined mixing is available only for REFIL")
        within, interaction = self.imagination_visibility(observation, group_a)
        context = torch.cat(
            [observation.self_obs, observation.task_obs], dim=-1
        )
        return self.mixer.imagined(
            within_q,
            interaction_q,
            state,
            context,
            observation.agent_mask,
            within,
            interaction,
        )

    @torch.no_grad()
    def act(
        self,
        observation: TeamObservation,
        hidden: Optional[Tensor] = None,
        epsilon: float = 0.0,
        last_action: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        q_values, next_hidden = self.agent_q(
            observation, hidden, last_action=last_action
        )
        greedy = q_values.argmax(dim=-1)
        if epsilon <= 0.0:
            return greedy, next_hidden
        random_scores = torch.rand_like(q_values).masked_fill(~observation.avail_actions.bool(), -1.0)
        random_actions = random_scores.argmax(dim=-1)
        explore = torch.rand_like(greedy, dtype=torch.float32) < epsilon
        actions = torch.where(explore, random_actions, greedy)
        actions = torch.where(observation.agent_mask.bool(), actions, torch.zeros_like(actions))
        return actions, next_hidden
