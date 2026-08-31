"""Whole-episode replay with padding for variable team and entity counts."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from open_score.contracts import GlobalState, TeamObservation
from open_score.stage1.curriculum import Scale


@dataclass(frozen=True)
class TeamEpisode:
    observations: Tuple[Dict[str, np.ndarray], ...]
    actions: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    scale: Scale
    side: str
    seed: int

    def __post_init__(self) -> None:
        steps = len(self.observations) - 1
        if steps < 1 or self.actions.shape[0] != steps:
            raise ValueError("an episode needs T actions and T+1 observations")
        if self.rewards.shape != (steps,) or self.done.shape != (steps,):
            raise ValueError("reward and done vectors must have length T")

    @property
    def length(self) -> int:
        return self.actions.shape[0]


@dataclass(frozen=True)
class CompetitiveEpisode:
    red: TeamEpisode
    blue: TeamEpisode
    outcome_red: float
    target_position: Tuple[float, float, float]
    red_policy_name: str
    blue_policy_name: str

    @property
    def length(self) -> int:
        return self.red.length


@dataclass
class PaddedEpisodeBatch:
    entity_obs: Tensor
    entity_mask: Tensor
    self_obs: Tensor
    task_obs: Tensor
    agent_mask: Tensor
    avail_actions: Tensor
    action_entity_index: Tensor
    action_target_type: Tensor
    state_entities: Tensor
    state_mask: Tensor
    actions: Tensor
    rewards: Tensor
    done: Tensor
    filled: Tensor
    scales: Tensor

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def max_steps(self) -> int:
        return self.actions.shape[1]

    def observation_at(self, time: int) -> TeamObservation:
        return TeamObservation(
            self.entity_obs[:, time],
            self.entity_mask[:, time],
            self.self_obs[:, time],
            self.task_obs[:, time],
            self.agent_mask[:, time],
            self.avail_actions[:, time],
            self.action_entity_index[:, time],
            self.action_target_type[:, time],
        )

    def state_at(self, time: int) -> GlobalState:
        return GlobalState(self.state_entities[:, time], self.state_mask[:, time])


def _copy_observation(
    destination: Dict[str, np.ndarray],
    batch_index: int,
    time_index: int,
    source: Dict[str, np.ndarray],
) -> None:
    agents, entities = source["entity_obs"].shape[:2]
    destination["entity_obs"][batch_index, time_index, :agents, :entities] = source["entity_obs"]
    destination["entity_mask"][batch_index, time_index, :agents, :entities] = source["entity_mask"]
    destination["self_obs"][batch_index, time_index, :agents] = source["self_obs"]
    destination["task_obs"][batch_index, time_index, :agents] = source["task_obs"]
    destination["agent_mask"][batch_index, time_index, :agents] = source["agent_mask"]
    destination["avail_actions"][batch_index, time_index, :agents] = source["avail_actions"]
    if "action_entity_index" in source:
        destination["action_entity_index"][batch_index, time_index, :agents] = source[
            "action_entity_index"
        ]
        destination["action_target_type"][batch_index, time_index, :agents] = source[
            "action_target_type"
        ]
    state_entities = source["state_entities"]
    destination["state_entities"][batch_index, time_index, : len(state_entities)] = state_entities
    destination["state_mask"][batch_index, time_index, : len(source["state_mask"])] = source["state_mask"]


def collate_episodes(
    episodes: Sequence[TeamEpisode],
    device: torch.device = torch.device("cpu"),
) -> PaddedEpisodeBatch:
    if not episodes:
        raise ValueError("cannot collate an empty episode list")
    first = episodes[0].observations[0]
    batch = len(episodes)
    max_steps = max(episode.length for episode in episodes)
    max_agents = max(observation["entity_obs"].shape[0] for episode in episodes for observation in episode.observations)
    max_entities = max(observation["entity_obs"].shape[1] for episode in episodes for observation in episode.observations)
    action_dim = first["avail_actions"].shape[-1]
    arrays = {
        "entity_obs": np.zeros((batch, max_steps + 1, max_agents, max_entities, first["entity_obs"].shape[-1]), np.float32),
        "entity_mask": np.zeros((batch, max_steps + 1, max_agents, max_entities), bool),
        "self_obs": np.zeros((batch, max_steps + 1, max_agents, first["self_obs"].shape[-1]), np.float32),
        "task_obs": np.zeros((batch, max_steps + 1, max_agents, first["task_obs"].shape[-1]), np.float32),
        "agent_mask": np.zeros((batch, max_steps + 1, max_agents), bool),
        "avail_actions": np.zeros((batch, max_steps + 1, max_agents, action_dim), bool),
        "action_entity_index": np.full(
            (batch, max_steps + 1, max_agents, action_dim), -1, np.int64
        ),
        "action_target_type": np.zeros(
            (batch, max_steps + 1, max_agents, action_dim), np.int64
        ),
        "state_entities": np.zeros((batch, max_steps + 1, max_entities, first["state_entities"].shape[-1]), np.float32),
        "state_mask": np.zeros((batch, max_steps + 1, max_entities), bool),
    }
    actions = np.zeros((batch, max_steps, max_agents), np.int64)
    rewards = np.zeros((batch, max_steps), np.float32)
    done = np.ones((batch, max_steps), np.float32)
    filled = np.zeros((batch, max_steps), np.float32)
    scales = np.zeros((batch, 2), np.int64)
    for batch_index, episode in enumerate(episodes):
        scales[batch_index] = episode.scale
        for time_index in range(max_steps + 1):
            source = episode.observations[min(time_index, episode.length)]
            _copy_observation(arrays, batch_index, time_index, source)
        agents = episode.actions.shape[1]
        actions[batch_index, : episode.length, :agents] = episode.actions
        rewards[batch_index, : episode.length] = episode.rewards
        done[batch_index, : episode.length] = episode.done
        filled[batch_index, : episode.length] = 1.0

    tensor = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=device)
    return PaddedEpisodeBatch(
        entity_obs=tensor(arrays["entity_obs"], torch.float32),
        entity_mask=tensor(arrays["entity_mask"], torch.bool),
        self_obs=tensor(arrays["self_obs"], torch.float32),
        task_obs=tensor(arrays["task_obs"], torch.float32),
        agent_mask=tensor(arrays["agent_mask"], torch.bool),
        avail_actions=tensor(arrays["avail_actions"], torch.bool),
        action_entity_index=tensor(arrays["action_entity_index"], torch.long),
        action_target_type=tensor(arrays["action_target_type"], torch.long),
        state_entities=tensor(arrays["state_entities"], torch.float32),
        state_mask=tensor(arrays["state_mask"], torch.bool),
        actions=tensor(actions, torch.long),
        rewards=tensor(rewards, torch.float32),
        done=tensor(done, torch.float32),
        filled=tensor(filled, torch.float32),
        scales=tensor(scales, torch.long),
    )


class EpisodeReplayBuffer:
    def __init__(self, capacity: int = 2_000, seed: Optional[int] = None):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.episodes: List[TeamEpisode] = []
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episode: TeamEpisode) -> None:
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
        else:
            self.episodes[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> PaddedEpisodeBatch:
        if not 1 <= batch_size <= len(self.episodes):
            raise ValueError("batch_size must be between one and current replay size")
        indices = self.rng.choice(len(self.episodes), size=batch_size, replace=False)
        return collate_episodes([self.episodes[int(index)] for index in indices], device)

    def sample_scale_balanced(
        self, batch_size: int, device: torch.device
    ) -> PaddedEpisodeBatch:
        """Stratify a replay batch across every currently represented scale.

        This is not prioritized replay: each scale gets an equal-size quota and
        episodes remain uniformly sampled within a scale.  When ``batch_size``
        is smaller than the number of represented scales, a rotating random
        subset is used.  Sampling remains without replacement.
        """

        if not 1 <= batch_size <= len(self.episodes):
            raise ValueError("batch_size must be between one and current replay size")
        by_scale: Dict[Scale, List[int]] = {}
        for index, episode in enumerate(self.episodes):
            by_scale.setdefault(episode.scale, []).append(index)
        scales = list(by_scale)
        self.rng.shuffle(scales)
        selected: List[int] = []
        while len(selected) < batch_size:
            made_progress = False
            for scale in scales:
                remaining = [index for index in by_scale[scale] if index not in selected]
                if not remaining:
                    continue
                selected.append(int(self.rng.choice(remaining)))
                made_progress = True
                if len(selected) == batch_size:
                    break
            if not made_progress:  # pragma: no cover - guarded by size validation
                raise AssertionError("could not fill a scale-balanced replay batch")
        return collate_episodes([self.episodes[index] for index in selected], device)
