"""HAD rollout collectors for the formal Stage-2 counterfactual protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch

from .canonical import HADCanonicalizer
from .data import Stage2Record, record_from_rollout, validate_counterfactual_design


PolicySpec = Union[str, Mapping[str, object]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _NoisyController:
    """Apply reproducible primitive-action noise without adding macro actions."""

    def __init__(self, controller, probability: float):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("action_noise must lie inside [0,1]")
        self.controller = controller
        self.probability = float(probability)
        self.name = controller.name

    def reset(self) -> None:
        self.controller.reset()

    def act(self, adapter, side, observation, rng) -> np.ndarray:
        actions = np.asarray(
            self.controller.act(adapter, side, observation, rng), dtype=np.int64
        ).copy()
        if self.probability <= 0.0:
            return actions
        for index, available in enumerate(observation["avail_actions"]):
            if rng.random() < self.probability:
                actions[index] = int(rng.choice(np.flatnonzero(available)))
        return actions


def _checkpoint_payload(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch 1.13 compatibility.
        return torch.load(path, map_location=device)


def _controller_from_spec(
    spec: PolicySpec,
    *,
    side: str,
    device: torch.device,
    scales: Sequence[Tuple[int, int]],
    max_steps: int,
    shaping_scale: float,
    strict_checkpoint_contract: bool = False,
):
    """Build a rule or frozen Stage-1 controller plus auditable metadata."""

    from open_score.envs import HADStage1Adapter
    from open_score.stage1 import (
        MAPPOController,
        QMixController,
        RuleBasedController,
        VariableScaleMAPPO,
        VariableScaleQMIX,
        VariableScaleVDN,
    )

    values: Dict[str, object] = (
        {"kind": "rule", "style": spec} if isinstance(spec, str) else dict(spec)
    )
    kind = str(values.get("kind", "rule"))
    action_noise = float(values.get("action_noise", 0.0))
    if kind == "rule":
        style = str(values["style"])
        controller = RuleBasedController(style)
        policy_id = str(values.get("policy_id", controller.name))
        controller.name = policy_id
        version = str(values.get("version", "rule-v1"))
        metadata = {
            "kind": "rule",
            "side": side,
            "policy_id": policy_id,
            "version": version,
            "style": style,
            "action_noise": action_noise,
            "checkpoint": None,
            "checkpoint_sha256": "",
        }
    elif kind == "checkpoint":
        checkpoint = Path(str(values["path"])).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"Stage-1 checkpoint does not exist: {checkpoint}")
        checkpoint_hash = _sha256(checkpoint)
        expected_hash = str(values.get("expected_sha256", "")).lower()
        if strict_checkpoint_contract and not expected_hash:
            raise ValueError("formal Stage-1 checkpoint spec requires expected_sha256")
        if expected_hash and (
            len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
        if expected_hash and checkpoint_hash.lower() != expected_hash:
            raise ValueError("Stage-1 checkpoint SHA-256 differs from the pinned digest")
        payload = _checkpoint_payload(checkpoint, device)
        if not isinstance(payload, Mapping):
            raise ValueError("Stage-1 checkpoint root must be a mapping")
        extra = dict(payload.get("extra", {}))
        required_metadata = {
            "algorithm",
            "train_side",
            "registered_scales",
            "protocol_version",
            "selection",
            "contract",
            "model_config",
            "environment_protocol",
            "reward_protocol",
        }
        missing_metadata = sorted(required_metadata - set(extra))
        if strict_checkpoint_contract and missing_metadata:
            raise ValueError(
                "formal Stage-1 checkpoint lacks required metadata: "
                + ", ".join(missing_metadata)
            )
        saved_algorithm = str(extra.get("algorithm", "")).lower()
        algorithm = str(values.get("algorithm", saved_algorithm)).lower()
        if algorithm not in {"qmix", "vdn", "mappo"}:
            raise ValueError("checkpoint policy algorithm must be qmix, vdn or mappo")
        if saved_algorithm and saved_algorithm != algorithm:
            raise ValueError("Stage-1 checkpoint algorithm metadata differs from policy spec")
        if str(extra.get("train_side", side)) != side:
            raise ValueError("Stage-1 checkpoint train_side does not match policy side")
        if strict_checkpoint_contract and not str(extra.get("protocol_version", "")).startswith(
            "had-stage1-"
        ):
            raise ValueError("formal checkpoint has an unrecognised Stage-1 protocol version")
        if strict_checkpoint_contract and str(extra.get("selection")) not in {
            "paired_eval_best",
            "validation_best",
        }:
            raise ValueError("formal Stage-2 collection requires a validation-selected checkpoint")
        environment = dict(extra.get("environment_protocol", {}))
        if environment.get("max_steps") is not None and int(environment["max_steps"]) != max_steps:
            raise ValueError("Stage-1 checkpoint max_steps differs from Stage-2 collector")
        registered = {
            tuple(map(int, scale)) for scale in extra.get("registered_scales", scales)
        }
        if not set(scales).issubset(registered):
            raise ValueError("Stage-2 requests a scale outside Stage-1 checkpoint metadata")
        reward = dict(extra.get("reward_protocol", {}))
        if reward.get("potential_shaping_scale") is not None and float(
            reward["potential_shaping_scale"]
        ) != shaping_scale:
            raise ValueError("Stage-1 checkpoint shaping scale differs from collector")
        model_config = dict(extra.get("model_config", {}))
        agent_hidden = int(
            model_config.get("agent_hidden_dim", values.get("agent_hidden_dim", 64))
        )
        critic_hidden = int(
            model_config.get(
                "critic_or_mixer_hidden_dim", values.get("critic_hidden_dim", 64)
            )
        )
        mixing_dim = int(model_config.get("mixing_dim", values.get("mixing_dim", 32)))
        dimensions = (
            HADStage1Adapter.ENTITY_DIM,
            HADStage1Adapter.SELF_DIM,
            HADStage1Adapter.TASK_DIM,
            HADStage1Adapter.STATE_ENTITY_DIM,
            HADStage1Adapter.ACTION_DIM,
        )
        saved_contract = dict(extra.get("contract", {}))
        encoder_kind = str(saved_contract.get("encoder_kind", "deepset"))
        attention_heads = int(saved_contract.get("attention_heads", 4))
        for key, expected in zip(
            ("entity_dim", "self_dim", "task_dim", "state_entity_dim", "action_dim"),
            dimensions,
        ):
            if saved_contract.get(key) is not None and int(saved_contract[key]) != expected:
                raise ValueError(f"Stage-1 checkpoint tensor contract mismatch: {key}")
        if algorithm == "qmix":
            model = VariableScaleQMIX(
                *dimensions,
                agent_hidden_dim=agent_hidden,
                mixer_hidden_dim=critic_hidden,
                mixing_dim=mixing_dim,
                encoder_kind=encoder_kind,
                attention_heads=attention_heads,
            ).to(device)
            model.load_state_dict(payload["online"])
            controller = QMixController(model, device, epsilon=0.0)
        elif algorithm == "vdn":
            model = VariableScaleVDN(
                *dimensions,
                agent_hidden_dim=agent_hidden,
                encoder_kind=encoder_kind,
                attention_heads=attention_heads,
            ).to(device)
            model.load_state_dict(payload["online"])
            controller = QMixController(model, device, epsilon=0.0)
        else:
            model = VariableScaleMAPPO(
                *dimensions,
                actor_hidden_dim=agent_hidden,
                critic_hidden_dim=critic_hidden,
                encoder_kind=encoder_kind,
                attention_heads=attention_heads,
            ).to(device)
            model.load_state_dict(payload["model"])
            controller = MAPPOController(model, device, deterministic=True)
        policy_id = str(values.get("policy_id", f"checkpoint:{algorithm}"))
        controller.name = policy_id
        version = str(values.get("version", f"sha256:{checkpoint_hash[:12]}"))
        metadata = {
            "kind": "checkpoint",
            "side": side,
            "policy_id": policy_id,
            "version": version,
            "algorithm": algorithm,
            "action_noise": action_noise,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "stage1_protocol_version": extra.get("protocol_version"),
            "stage1_selection": extra.get("selection"),
            "encoder_kind": encoder_kind,
            "attention_heads": attention_heads,
        }
    else:
        raise ValueError("policy kind must be rule or checkpoint")
    if action_noise:
        controller = _NoisyController(controller, action_noise)
    return controller, metadata


def _run_had_continuation(
    factory,
    scale: Tuple[int, int],
    red_controller,
    blue_controller,
    *,
    snapshot,
    continuation_seed: int,
    rollout_horizon_steps: int,
):
    """Restore one physical root and run a bounded, CRN-seeded continuation."""

    from open_score.stage1.replay import CompetitiveEpisode, TeamEpisode

    adapter = factory.get(scale)
    observations = adapter.restore(snapshot, continuation_seed=int(continuation_seed))
    red_controller.reset()
    blue_controller.reset()
    # Side-specific streams keep Blue threat noise common across Red policies.
    red_rng = np.random.default_rng(int(continuation_seed) + 1_000_003)
    blue_rng = np.random.default_rng(int(continuation_seed) + 2_000_003)
    red_observations = [observations["Red"]]
    blue_observations = [observations["Blue"]]
    red_actions, blue_actions, red_rewards, blue_rewards, done_flags = [], [], [], [], []
    done = False
    info: Dict[str, object] = {}
    while not done and len(done_flags) < rollout_horizon_steps:
        action_red = red_controller.act(adapter, "Red", observations["Red"], red_rng)
        action_blue = blue_controller.act(adapter, "Blue", observations["Blue"], blue_rng)
        observations, rewards, done, info = adapter.step(action_red, action_blue)
        red_actions.append(action_red)
        blue_actions.append(action_blue)
        red_rewards.append(rewards["Red"])
        blue_rewards.append(rewards["Blue"])
        done_flags.append(float(done))
        red_observations.append(observations["Red"])
        blue_observations.append(observations["Blue"])
    red = TeamEpisode(
        tuple(red_observations),
        np.stack(red_actions),
        np.asarray(red_rewards, np.float32),
        np.asarray(done_flags, np.float32),
        scale,
        "Red",
        int(continuation_seed),
    )
    blue = TeamEpisode(
        tuple(blue_observations),
        np.stack(blue_actions),
        np.asarray(blue_rewards, np.float32),
        np.asarray(done_flags, np.float32),
        scale,
        "Blue",
        int(continuation_seed),
    )
    return CompetitiveEpisode(
        red,
        blue,
        float(info["outcome_red"]),
        tuple(float(value) for value in info["target_position"]),
        red_controller.name,
        blue_controller.name,
    )


def _had_prefix_snapshots(
    factory,
    scale: Tuple[int, int],
    *,
    root_seed: int,
    snapshot_steps: Sequence[int],
    red_controller,
    blue_controller,
):
    """Generate branch points and explain every snapshot lost to a terminal prefix."""

    adapter = factory.get(scale)
    np.random.seed(int(root_seed) % (2**32))
    observations = adapter.reset(seed=int(root_seed))
    red_controller.reset()
    blue_controller.reset()
    red_rng = np.random.default_rng(int(root_seed) + 3_000_017)
    blue_rng = np.random.default_rng(int(root_seed) + 4_000_037)
    requested = set(map(int, snapshot_steps))
    snapshots = {}
    terminal_step = None
    terminal_outcome_red = None
    if 0 in requested:
        snapshots[0] = (adapter.snapshot(), observations)
    for step in range(1, max(requested, default=0) + 1):
        action_red = red_controller.act(adapter, "Red", observations["Red"], red_rng)
        action_blue = blue_controller.act(adapter, "Blue", observations["Blue"], blue_rng)
        observations, _, done, info = adapter.step(action_red, action_blue)
        # A terminal state is already labelled and cannot be a decision root.
        if step in requested and not done:
            snapshots[step] = (adapter.snapshot(), observations)
        if done:
            terminal_step = step
            if info.get("outcome_red") is not None:
                terminal_outcome_red = float(info["outcome_red"])
            break
    missing = sorted(requested - set(snapshots))
    terminal_attrition = [
        step
        for step in missing
        if terminal_step is not None and int(terminal_step) <= int(step)
    ]
    unexplained_missing = sorted(set(missing) - set(terminal_attrition))
    return snapshots, {
        "prefix_terminal": terminal_step is not None,
        "prefix_terminal_step": terminal_step,
        "prefix_terminal_outcome_red": terminal_outcome_red,
        "requested_snapshot_steps": sorted(requested),
        "realized_snapshot_steps": sorted(snapshots),
        "terminal_attrition_snapshot_steps": terminal_attrition,
        "unexplained_missing_snapshot_steps": unexplained_missing,
    }


def record_from_had_episode(
    episode,
    *,
    lineage_group_id: str,
    root_id: str,
    rollout_id: str,
    seed: int,
    scenario_id: str,
    horizon_steps: int,
    command_steps: int,
    defender_policy_version: str,
    attacker_policy_version: str,
    capability_version: str,
    continuation_id: str = "continuation-000",
    stage1_checkpoint_sha256: str = "",
    root_seed: Optional[int] = None,
    stochasticity_profile: str = "had-crn-v1",
) -> Stage2Record:
    """Convert a HAD ``CompetitiveEpisode`` to the formal S2 contract."""

    defender_count, attacker_count = map(int, episode.red.scale)
    initial = episode.red.observations[0]
    state_trace = [observation["state_entities"] for observation in episode.red.observations]
    initial_entities = state_trace[0]
    final_entities = state_trace[-1]
    target_rows = final_entities[:, 10] > 0.5
    attacker_rows = final_entities[:, 9] > 0.5
    target_alive = bool(final_entities[target_rows, 7].item() > 0.5)
    attackers_alive = bool(np.any(final_entities[attacker_rows, 7] > 0.5))
    # Column 11 is the normalized remaining HAD episode horizon.  Reaching
    # zero is a natural, observed defender terminal under the Stage-1
    # contract; exhausting only the local continuation budget is instead an
    # administrative right-censoring event.
    natural_horizon_reached = bool(
        final_entities.shape[1] > 11
        and final_entities[target_rows, 11].item() <= 1e-8
    )
    if not target_alive:
        outcome = "breach"
    elif not attackers_alive:
        outcome = "defender_win"
    elif natural_horizon_reached:
        outcome = "defender_win"
    elif episode.length >= horizon_steps:
        outcome = "timeout"  # administrative right-censoring
    else:
        outcome = "defender_win"
    target_health = [
        float(state[state[:, 10] > 0.5, 6].item()) for state in state_trace
    ]
    initial_red_health = float(initial_entities[initial_entities[:, 8] > 0.5, 6].sum())
    initial_blue_health = float(initial_entities[initial_entities[:, 9] > 0.5, 6].sum())
    final_red_health = float(final_entities[final_entities[:, 8] > 0.5, 6].sum())
    final_blue_health = float(final_entities[final_entities[:, 9] > 0.5, 6].sum())
    defender_survivors = int(
        np.sum(final_entities[final_entities[:, 8] > 0.5, 7] > 0.5)
    )
    attacker_survivors = int(
        np.sum(final_entities[final_entities[:, 9] > 0.5, 7] > 0.5)
    )
    initial_defender_survivors = int(
        np.sum(initial_entities[initial_entities[:, 8] > 0.5, 7] > 0.5)
    )
    initial_attacker_survivors = int(
        np.sum(initial_entities[initial_entities[:, 9] > 0.5, 7] > 0.5)
    )
    threat_distances = []
    for state in state_trace:
        target_position = state[state[:, 10] > 0.5, :3][0]
        alive_blue = state[(state[:, 9] > 0.5) & (state[:, 7] > 0.5), :3]
        if len(alive_blue):
            threat_distances.append(
                float(np.linalg.norm(alive_blue - target_position, axis=1).min())
            )
    from open_score.envs.had_stage1 import ACCELERATION_PRIMITIVES

    red_action_cost = float(
        sum(np.linalg.norm(ACCELERATION_PRIMITIVES[actions], axis=-1).sum() for actions in episode.red.actions)
    )
    blue_action_cost = float(
        sum(np.linalg.norm(ACCELERATION_PRIMITIVES[actions], axis=-1).sum() for actions in episode.blue.actions)
    )
    candidate_id = f"{episode.red_policy_name}@{defender_policy_version}"
    threat_id = f"{episode.blue_policy_name}@{attacker_policy_version}"
    return record_from_rollout(
        canonical_state=HADCanonicalizer().from_observation(initial),
        outcome=outcome,
        terminal_steps=episode.length,
        environment_id="HAD",
        scenario_id=scenario_id,
        lineage_group_id=lineage_group_id,
        root_id=root_id,
        rollout_id=rollout_id,
        seed=int(seed),
        defender_count=defender_count,
        attacker_count=attacker_count,
        defender_policy_id=episode.red_policy_name,
        attacker_policy_id=episode.blue_policy_name,
        defender_policy_version=defender_policy_version,
        attacker_policy_version=attacker_policy_version,
        capability_version=capability_version,
        horizon_steps=horizon_steps,
        command_steps=command_steps,
        candidate_id=candidate_id,
        threat_id=threat_id,
        continuation_id=continuation_id,
        stage1_checkpoint_sha256=stage1_checkpoint_sha256,
        root_seed=int(seed if root_seed is None else root_seed),
        continuation_seed=int(seed),
        stochasticity_profile=stochasticity_profile,
        # Historical field name retained for schema compatibility.  This is a
        # fully observed local-window safety utility: +1 iff the asset survives
        # the observation window (including an administratively censored
        # continuation), not an uncensored eventual-game payoff.
        payoff_red=(-1.0 if outcome == "breach" else 1.0),
        target_final_health_fraction=target_health[-1],
        target_min_health_fraction=min(target_health),
        defender_survivors=defender_survivors,
        attacker_survivors=attacker_survivors,
        defender_casualties=initial_defender_survivors - defender_survivors,
        attacker_casualties=initial_attacker_survivors - attacker_survivors,
        breach_steps=(episode.length if outcome == "breach" else None),
        # A natural-horizon defender success does not imply that all attackers
        # were neutralized.  Keep the event-time label tied to the physical
        # event rather than the broader defender-win class.
        attackers_neutralized_steps=(episode.length if not attackers_alive else None),
        minimum_threat_distance=(min(threat_distances) if threat_distances else np.sqrt(3.0)),
        cumulative_target_damage=max(0.0, target_health[0] - target_health[-1]),
        cumulative_defender_damage=max(0.0, initial_red_health - final_red_health),
        cumulative_attacker_damage=max(0.0, initial_blue_health - final_blue_health),
        red_action_cost=red_action_cost,
        blue_action_cost=blue_action_cost,
    )


def iter_had_records(
    *,
    scales: Sequence[Tuple[int, int]],
    seeds: Optional[Iterable[int]] = None,
    seeds_by_scale: Optional[Mapping[str, Sequence[int]]] = None,
    defender_policies: Sequence[str] = ("guard", "intercept"),
    attacker_policies: Sequence[str] = ("rush", "split_rush"),
    defender_candidates: Optional[Sequence[PolicySpec]] = None,
    attacker_threats: Optional[Sequence[PolicySpec]] = None,
    continuations_per_cell: int = 2,
    continuation_seed_stride: int = 1_000_000,
    max_steps: int = 80,
    snapshot_steps: Sequence[int] = (0,),
    rollout_horizon_steps: Optional[int] = None,
    prefix_defender_policy: str = "guard",
    prefix_attacker_policy: str = "rush",
    command_steps: int = 20,
    capability_version: str = "had-stage1-v2",
    policy_version: str = "rule-v1",
    shaping_scale: float = 0.5,
    device: Union[str, torch.device] = "cpu",
    strict_checkpoint_contract: bool = False,
    collection_audit: Optional[MutableMapping[str, object]] = None,
) -> Iterator[Stage2Record]:
    """Collect root-matched Red candidates against exogenous Blue threats.

    ``root_seed`` determines only the physical initial state.  ``continuation``
    determines policy/environment random numbers and is shared by every Red
    candidate under a fixed root and Blue threat (common random numbers).
    """

    from open_score.stage1 import HADStage1Factory

    if not scales:
        raise ValueError("HAD collection needs at least one scale")
    if any(int(red) <= int(blue) for red, blue in scales):
        raise ValueError("formal HAD protocol requires strict Red superiority")
    if (seeds is None) == (seeds_by_scale is None):
        raise ValueError("provide exactly one of seeds or seeds_by_scale")
    if continuations_per_cell < 2:
        raise ValueError("formal collection requires at least two continuations per cell")
    snapshot_steps = tuple(sorted(set(map(int, snapshot_steps))))
    if not snapshot_steps or snapshot_steps[0] < 0:
        raise ValueError("snapshot_steps must contain non-negative command times")
    rollout_horizon = int(
        max_steps if rollout_horizon_steps is None else rollout_horizon_steps
    )
    if rollout_horizon < 1 or snapshot_steps[-1] + rollout_horizon > max_steps:
        raise ValueError(
            "every snapshot needs a full rollout_horizon_steps before max_steps"
        )
    if not 1 <= command_steps <= rollout_horizon:
        raise ValueError("command_steps must lie inside rollout_horizon_steps")
    target_device = torch.device(device)
    red_specs: Sequence[PolicySpec] = defender_candidates or tuple(
        {"kind": "rule", "style": policy, "version": policy_version}
        for policy in defender_policies
    )
    blue_specs: Sequence[PolicySpec] = attacker_threats or tuple(
        {"kind": "rule", "style": policy, "version": policy_version}
        for policy in attacker_policies
    )
    if not red_specs or not blue_specs:
        raise ValueError("HAD collection needs Red candidates and Blue threats")
    factory = HADStage1Factory(max_steps=max_steps, shaping_scale=shaping_scale)
    red_controllers = [
        _controller_from_spec(
            spec,
            side="Red",
            device=target_device,
            scales=scales,
            max_steps=max_steps,
            shaping_scale=shaping_scale,
            strict_checkpoint_contract=strict_checkpoint_contract,
        )
        for spec in red_specs
    ]
    blue_controllers = [
        _controller_from_spec(
            spec,
            side="Blue",
            device=target_device,
            scales=scales,
            max_steps=max_steps,
            shaping_scale=shaping_scale,
            strict_checkpoint_contract=strict_checkpoint_contract,
        )
        for spec in blue_specs
    ]
    prefix_red, _ = _controller_from_spec(
        {"kind": "rule", "style": prefix_defender_policy},
        side="Red",
        device=target_device,
        scales=scales,
        max_steps=max_steps,
        shaping_scale=shaping_scale,
    )
    prefix_blue, _ = _controller_from_spec(
        {"kind": "rule", "style": prefix_attacker_policy},
        side="Blue",
        device=target_device,
        scales=scales,
        max_steps=max_steps,
        shaping_scale=shaping_scale,
    )
    common_seeds = None if seeds is None else tuple(int(seed) for seed in seeds)
    if collection_audit is not None:
        collection_audit.clear()
        collection_audit.update(
            {
                "status": "recording",
                "requested_snapshot_steps": list(snapshot_steps),
                "planned_snapshot_roots": 0,
                "realized_snapshot_roots": 0,
                "terminal_attrition_roots": 0,
                "unexplained_missing_roots": 0,
                "snapshot_zero_expected_roots": 0,
                "snapshot_zero_realized_roots": 0,
                "red_candidate_count": len(red_controllers),
                "blue_threat_count": len(blue_controllers),
                "continuations_per_cell": continuations_per_cell,
                "by_scale_snapshot": {},
                "missing_snapshot_roots": [],
            }
        )
    for scale in scales:
        defender_count, attacker_count = map(int, scale)
        scale_name = f"{defender_count}v{attacker_count}"
        root_seeds = common_seeds if common_seeds is not None else tuple(
            int(seed) for seed in seeds_by_scale[scale_name]
        )
        if collection_audit is not None:
            per_scale = collection_audit["by_scale_snapshot"].setdefault(
                scale_name, {}
            )
            for snapshot_step in snapshot_steps:
                per_scale[f"snapshot-{snapshot_step:03d}"] = {
                    "planned": 0,
                    "realized": 0,
                    "terminal_attrition": 0,
                    "unexplained_missing": 0,
                }
        for root_seed in root_seeds:
            lineage_group_id = f"had:master-seed-{root_seed}"
            roots, prefix_audit = _had_prefix_snapshots(
                factory,
                (defender_count, attacker_count),
                root_seed=root_seed,
                snapshot_steps=snapshot_steps,
                red_controller=prefix_red,
                blue_controller=prefix_blue,
            )
            if collection_audit is not None:
                for snapshot_step in snapshot_steps:
                    bucket = collection_audit["by_scale_snapshot"][scale_name][
                        f"snapshot-{snapshot_step:03d}"
                    ]
                    bucket["planned"] += 1
                    collection_audit["planned_snapshot_roots"] += 1
                    if snapshot_step == 0:
                        collection_audit["snapshot_zero_expected_roots"] += 1
                    if snapshot_step in roots:
                        bucket["realized"] += 1
                        collection_audit["realized_snapshot_roots"] += 1
                        if snapshot_step == 0:
                            collection_audit["snapshot_zero_realized_roots"] += 1
                        continue
                    terminal_attrition = snapshot_step in set(
                        prefix_audit["terminal_attrition_snapshot_steps"]
                    )
                    status = (
                        "terminal_prefix_attrition"
                        if terminal_attrition
                        else "unexplained_missing"
                    )
                    bucket[
                        "terminal_attrition"
                        if terminal_attrition
                        else "unexplained_missing"
                    ] += 1
                    collection_audit[
                        "terminal_attrition_roots"
                        if terminal_attrition
                        else "unexplained_missing_roots"
                    ] += 1
                    collection_audit["missing_snapshot_roots"].append(
                        {
                            "lineage_group_id": lineage_group_id,
                            "root_seed": int(root_seed),
                            "scale": scale_name,
                            "snapshot_step": int(snapshot_step),
                            "status": status,
                            "prefix_terminal_step": prefix_audit[
                                "prefix_terminal_step"
                            ],
                            "prefix_terminal_outcome_red": prefix_audit[
                                "prefix_terminal_outcome_red"
                            ],
                        }
                    )
            for snapshot_step, (snapshot, snapshot_observations) in roots.items():
                canonical_root = HADCanonicalizer().from_observation(
                    snapshot_observations["Red"]
                )
                state_hash = hashlib.sha256(canonical_root.tobytes()).hexdigest()[:16]
                root_id = (
                    f"{lineage_group_id}:{scale_name}:snapshot-{snapshot_step:03d}:"
                    f"state-{state_hash}"
                )
                for threat_index, (blue, blue_meta) in enumerate(blue_controllers):
                    for continuation_index in range(continuations_per_cell):
                        continuation_id = f"continuation-{continuation_index:03d}"
                        continuation_seed = (
                            int(root_seed) * continuation_seed_stride
                            + snapshot_step * 1009
                            + threat_index * 10_007
                            + continuation_index
                        )
                        for red, red_meta in red_controllers:
                            episode = _run_had_continuation(
                                factory,
                                (defender_count, attacker_count),
                                red,
                                blue,
                                snapshot=snapshot,
                                continuation_seed=continuation_seed,
                                rollout_horizon_steps=rollout_horizon,
                            )
                            candidate_id = f"{red_meta['policy_id']}@{red_meta['version']}"
                            threat_id = f"{blue_meta['policy_id']}@{blue_meta['version']}"
                            rollout_id = (
                                f"{root_id}:candidate={candidate_id}:threat={threat_id}:"
                                f"{continuation_id}"
                            )
                            stochasticity_profile = (
                                "had-crn-v2:"
                                f"red_action_noise={float(red_meta.get('action_noise', 0.0))}:"
                                f"blue_action_noise={float(blue_meta.get('action_noise', 0.0))}:"
                                "environment_rng=continuation_seed"
                            )
                            yield record_from_had_episode(
                                episode,
                                lineage_group_id=lineage_group_id,
                                root_id=root_id,
                                rollout_id=rollout_id,
                                seed=continuation_seed,
                                root_seed=root_seed,
                                scenario_id=(
                                    f"one_target_{scale_name}_snapshot_{snapshot_step:03d}"
                                ),
                                defender_policy_version=str(red_meta["version"]),
                                attacker_policy_version=str(blue_meta["version"]),
                                capability_version=capability_version,
                                horizon_steps=rollout_horizon,
                                command_steps=command_steps,
                                continuation_id=continuation_id,
                                stage1_checkpoint_sha256=str(
                                    red_meta.get("checkpoint_sha256", "")
                                ),
                                stochasticity_profile=stochasticity_profile,
                            )
    if collection_audit is not None:
        collection_audit["expected_records_from_realized_roots"] = int(
            collection_audit["realized_snapshot_roots"]
            * len(red_controllers)
            * len(blue_controllers)
            * continuations_per_cell
        )
        collection_audit["status"] = "recorded"


def collect_had_records(**kwargs: object) -> List[Stage2Record]:
    """Materialise the iterator for small experiments and unit tests.

    Formal collection should use :func:`iter_had_records` so hundreds of
    thousands of JSON rows never coexist as Python objects in RAM.
    """

    records = list(iter_had_records(**kwargs))
    continuations_per_cell = int(kwargs.get("continuations_per_cell", 2))
    validate_counterfactual_design(
        records,
        min_continuations_per_cell=continuations_per_cell,
        require_common_random_numbers=True,
    )
    return records


__all__ = ["collect_had_records", "iter_had_records", "record_from_had_episode"]
