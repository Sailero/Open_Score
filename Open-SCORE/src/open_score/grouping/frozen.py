"""Immutable REFIL-QMIX inference with recurrent state owned by physical IDs."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch

from open_score.contracts import TeamObservation
from open_score.stage1.entity_qmix import VariableScaleQMIX

from .domain import Grouping


DEFAULT_STAGE1_PATH = Path(__file__).resolve().parents[3] / "assets" / "frozen" / "lcl.pt"


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _load_model(path: str, device: str, size: int, modified_ns: int):
    del size, modified_ns
    # Constructing a frozen model must not consume the upper policy's RNG.
    with torch.random.fork_rng(devices=[]):
        model = VariableScaleQMIX(
            entity_dim=12, self_dim=10, task_dim=7, state_entity_dim=12,
            action_dim=27, agent_hidden_dim=64, mixer_hidden_dim=128,
            mixing_dim=32, encoder_kind="refil", attention_heads=4,
            attention_embed_dim=128, hypernet_hidden_dim=128)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("extra", {}).get("architecture") != "REFIL-QMIX-HAD-v1":
        raise ValueError("Frozen LCL checkpoint must have REFIL-QMIX-HAD-v1 architecture")
    model.load_state_dict(payload["online"], strict=True)
    model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_stage1(path: Path | str | None = None, device: str = "cpu"):
    checkpoint = (DEFAULT_STAGE1_PATH if path is None else Path(path)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen LCL checkpoint is missing: {checkpoint}")
    stat = checkpoint.stat()
    return _load_model(str(checkpoint), str(device), stat.st_size, stat.st_mtime_ns)


class FrozenExecutor:
    """Group-local Red observation, all Blue visible, one shared physical world.

    Each live Red identity carries its GRU state and last physical action
    across group changes and decision events. Death removes only that ID.
    Reserve navigation is a public deterministic guard rule; it preserves
    latent memory while recording the actual rule action as last action.
    """

    def __init__(self, stage1_path: Path | str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = load_stage1(stage1_path, str(self.device))
        self.hidden: dict[int, np.ndarray] = {}
        self.last_actions: dict[int, int] = {}

    @property
    def hidden_dim(self) -> int:
        return self.model.agent.hidden_dim

    def reset(self, ids):
        self.hidden = {int(i): np.zeros(self.hidden_dim, dtype=np.float32) for i in ids}
        self.last_actions = {int(i): -1 for i in ids}

    def prune(self, ids):
        live = set(ids)
        self.hidden = {i: row for i, row in self.hidden.items() if i in live}
        self.last_actions = {i: value for i, value in self.last_actions.items() if i in live}

    def memory(self) -> dict[int, tuple[float, ...]]:
        return {i: tuple(map(float, row)) for i, row in self.hidden.items()}

    def snapshot(self) -> dict:
        return {"hidden": {i: row.copy() for i, row in self.hidden.items()},
                "last_actions": dict(self.last_actions)}

    def restore(self, snapshot: dict):
        self.hidden = {int(i): np.asarray(row, dtype=np.float32).copy()
                       for i, row in snapshot["hidden"].items()}
        self.last_actions = {int(i): int(value) for i, value in snapshot["last_actions"].items()}

    @staticmethod
    def _nearest_action(adapter, direction) -> int:
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        return 0 if norm < 1e-10 else int(np.argmax(adapter.action_vectors @ (direction / norm)))

    def reserve_actions(self, adapter, grouping: Grouping) -> dict[int, int]:
        reds, blues, targets = (adapter.agent_states("Red"), adapter.agent_states("Blue"),
                                adapter.target_states())
        coverage = {t: sum(len(group.members) for group in grouping.groups if group.target == t)
                    for t in targets}
        threat = {t: sum(1.0 / (1.0 + np.linalg.norm(blue["position"] - target["position"]) / 500.0)
                        for blue in blues.values() if blue["alive"])
                  for t, target in targets.items()}
        result = {}
        for i in grouping.reserve:
            position = reds[i]["position"]
            target_id = min(targets, key=lambda t: (
                -threat[t] / (1.0 + coverage[t]),
                float(np.linalg.norm(position - targets[t]["position"])), t))
            coverage[target_id] += 1
            destination = targets[target_id]["position"] + np.asarray([350.0, 0.0, 0.0])
            result[i] = self._nearest_action(adapter, destination - position)
        return result

    @torch.no_grad()
    def act(self, adapter, grouping: Grouping) -> dict[int, int]:
        red_ids = tuple(i for i, row in adapter.agent_states("Red").items() if row["alive"])
        blue_ids = tuple(i for i, row in adapter.agent_states("Blue").items() if row["alive"])
        self.prune(red_ids)
        grouping = grouping.prune(red_ids)
        grouping.validate(red_ids, adapter.target_ids)
        result = self.reserve_actions(adapter, grouping)
        observations = [adapter.local_observation("Red", group.target, group.members,
                                                   blue_ids, local_step=adapter.step_count)
                        for group in grouping.groups]
        if observations:
            batch = len(observations)
            max_agents = max(len(group.members) for group in grouping.groups)
            max_entities = max(row["entity_obs"].shape[1] for row in observations)
            entity_obs = np.zeros((batch, max_agents, max_entities, 12), np.float32)
            entity_mask = np.zeros((batch, max_agents, max_entities), bool)
            self_obs = np.zeros((batch, max_agents, 10), np.float32)
            task_obs = np.zeros((batch, max_agents, 7), np.float32)
            agent_mask = np.zeros((batch, max_agents), bool)
            avail = np.zeros((batch, max_agents, 27), bool)
            avail[..., 0] = True
            hidden = np.zeros((batch, max_agents, self.hidden_dim), np.float32)
            previous = np.zeros((batch, max_agents, 27), np.float32)
            for k, (group, row) in enumerate(zip(grouping.groups, observations)):
                agents, entities = row["entity_obs"].shape[:2]
                entity_obs[k, :agents, :entities] = row["entity_obs"]
                entity_mask[k, :agents, :entities] = row["entity_mask"]
                # The encoder also runs over padded agents: give each a safe
                # entity to attend to, while keeping its agent_mask false.
                entity_mask[k, agents:, 0] = True
                self_obs[k, :agents], task_obs[k, :agents] = row["self_obs"], row["task_obs"]
                agent_mask[k, :agents], avail[k, :agents] = row["agent_mask"], row["avail_actions"]
                for j, i in enumerate(group.members):
                    hidden[k, j] = self.hidden.setdefault(i, np.zeros(self.hidden_dim, np.float32))
                    old_action = self.last_actions.get(i, -1)
                    if old_action >= 0:
                        previous[k, j, old_action] = 1.0
            tensor = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=self.device)
            team = TeamObservation(tensor(entity_obs, torch.float32), tensor(entity_mask, torch.bool),
                                   tensor(self_obs, torch.float32), tensor(task_obs, torch.float32),
                                   tensor(agent_mask, torch.bool), tensor(avail, torch.bool))
            selected, next_hidden = self.model.act(team, tensor(hidden, torch.float32),
                                                   epsilon=0.0, last_action=tensor(previous, torch.float32))
            selected, next_hidden = selected.cpu().numpy(), next_hidden.cpu().numpy()
            for k, group in enumerate(grouping.groups):
                for j, i in enumerate(group.members):
                    result[i] = int(selected[k, j])
                    self.hidden[i] = next_hidden[k, j].copy()
        self.last_actions.update(result)
        return {int(i): int(result.get(i, 0)) for i in adapter.red_ids}
