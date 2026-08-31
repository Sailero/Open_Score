"""Canonical fixed-width local state encoders used by Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class HADCanonicalizer:
    """Convert HAD global entities into a target-centred 4v4 vector.

    HAD's state entity schema is ``position(3), velocity(3), health, alive,
    red, blue, target, remaining_horizon``.  Agents are sorted separately by
    distance to the target and lexicographic state features.  Consequently,
    permuting environment entity rows cannot change the result.

    Slots retain a ``present`` bit distinct from ``alive``.  A destroyed
    roster member is therefore distinguishable from padding, which is
    essential for command-time casualty evaluation.
    """

    max_defenders: int = 4
    max_attackers: int = 4
    entity_dim: int = 12

    @property
    def state_dim(self) -> int:
        # target: pos/vel/health/alive (8); agent: rel-pos/vel/health/alive/present (9)
        # final four values: roster and alive fractions for both sides.
        # One final scalar carries normalized remaining horizon from the S1
        # Markov state. Legacy 11-column rows are migrated with value 1.0.
        return 8 + 9 * (self.max_defenders + self.max_attackers) + 4 + 1

    @staticmethod
    def _sort_agents(agents: np.ndarray, target: np.ndarray) -> np.ndarray:
        if len(agents) < 2:
            return agents
        relative = agents[:, :3] - target[:3]
        distance = np.linalg.norm(relative, axis=1)
        # lexsort uses the final key as primary; feature keys resolve equal distances.
        keys = tuple(agents[:, index] for index in range(agents.shape[1] - 1, -1, -1))
        order = np.lexsort(keys + (distance,))
        return agents[order]

    @staticmethod
    def _agent_slots(agents: np.ndarray, target: np.ndarray, capacity: int) -> np.ndarray:
        slots = np.zeros((capacity, 9), dtype=np.float32)
        for index, agent in enumerate(agents):
            slots[index, :3] = agent[:3] - target[:3]
            slots[index, 3:6] = agent[3:6] - target[3:6]
            slots[index, 6] = agent[6]
            slots[index, 7] = agent[7]
            slots[index, 8] = 1.0
        return slots.reshape(-1)

    def __call__(
        self,
        state_entities: np.ndarray,
        state_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        entities = np.asarray(state_entities, dtype=np.float32)
        if entities.ndim != 2 or entities.shape[1] not in {11, self.entity_dim}:
            raise ValueError(
                f"HAD state_entities must have 11 legacy or {self.entity_dim} current columns"
            )
        if not np.all(np.isfinite(entities)):
            raise ValueError("HAD state_entities contain non-finite values")
        if state_mask is not None:
            mask = np.asarray(state_mask)
            if mask.shape != (len(entities),):
                raise ValueError("state_mask must have one value per state entity")
            if mask.dtype != np.bool_:
                raise ValueError("state_mask must be boolean")
            entities = entities[mask]
            if len(entities) == 0:
                raise ValueError("state_mask removes every state entity")
        if entities.shape[1] == 11:
            entities = np.concatenate(
                (entities, np.ones((len(entities), 1), dtype=np.float32)), axis=1
            )

        defender_rows = entities[:, 8] > 0.5
        attacker_rows = entities[:, 9] > 0.5
        target_rows = entities[:, 10] > 0.5
        if int(target_rows.sum()) != 1:
            raise ValueError("canonical HAD state requires exactly one target")
        defenders = entities[defender_rows]
        attackers = entities[attacker_rows]
        if not 1 <= len(defenders) <= self.max_defenders:
            raise ValueError("HAD defender roster exceeds the registered local capacity")
        if not 1 <= len(attackers) <= self.max_attackers:
            raise ValueError("HAD attacker roster exceeds the registered local capacity")
        target = entities[target_rows][0]
        defenders = self._sort_agents(defenders, target)
        attackers = self._sort_agents(attackers, target)
        counts = np.asarray(
            [
                len(defenders) / self.max_defenders,
                len(attackers) / self.max_attackers,
                float((defenders[:, 7] > 0.5).sum()) / self.max_defenders,
                float((attackers[:, 7] > 0.5).sum()) / self.max_attackers,
            ],
            dtype=np.float32,
        )
        result = np.concatenate(
            [
                target[:8],
                self._agent_slots(defenders, target, self.max_defenders),
                self._agent_slots(attackers, target, self.max_attackers),
                counts,
                np.asarray([target[11]], dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)
        if result.shape != (self.state_dim,):
            raise RuntimeError("internal HAD canonical-state dimension error")
        return result

    def from_observation(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        if "state_entities" not in observation:
            raise ValueError("HAD observation lacks state_entities")
        return self(observation["state_entities"], observation.get("state_mask"))
