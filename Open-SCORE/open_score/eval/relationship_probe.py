"""Frozen, scene-disjoint main0921 relationship probe on shared public trajectories."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
import fcntl
import gzip
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .experiment import PROBE_CONFIGS, atomic_json, checkpoint_info, run_directory

LABELS = ("nearest_target_distance", "nearest_other_defender_distance",
          "nearest_target_region_imbalance")
ALPHAS = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)
FIELDS = ("model_seed", "control", "label", "round", "split", "config", "n", "r2", "alpha")


class ProbeStopped(Exception):
    pass


def _check_stop(stop_requested):
    if stop_requested is not None and stop_requested():
        raise ProbeStopped()


def _progress(callback, **data):
    if callback is not None:
        callback(data)


@contextmanager
def _lock(path, stop_requested=None, timeout=60):
    """OS locks release on crash; a waiting worker returns instead of hanging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        deadline = time.monotonic() + timeout
        while True:
            _check_stop(stop_requested)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Probe worker is using {path.name}; retry after its progress advances")
                time.sleep(.5)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def scene_split(config_index, episode_index):
    """Scene groups are disjoint and both behavior policies enter every split."""
    local = episode_index % 50
    if config_index >= 2:
        return "ood_explore" if local < 25 else "ood_confirm"
    if local < 25:
        return "id_train"
    validation_end = 38 if episode_index < 50 else 37
    return "id_val" if local < validation_end else "id_test"


def scene_jobs():
    return [dict(config=list(config), config_index=ci, episode_index=ei,
                 episode_seed=200000 + ci * 100 + ei,
                 behavior="regir" if ei < 50 else "refil", split=scene_split(ci, ei))
            for ci, config in enumerate(PROBE_CONFIGS) for ei in range(100)]


def _scene_name(job):
    return f"scene_{job['episode_seed']}"


def _config_name(config):
    return f"{config[0]}v{config[1]}_K{config[2]}"


def _write_trajectory(path, payload):
    temporary = path.with_suffix(path.suffix + ".pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
    os.replace(temporary, path)


def _read_trajectory(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _verify_seed_separation(output):
    """Inspect recorded environment seeds, including sibling source versions."""
    from open_score.utils.logging import iter_records
    reserved = {j["episode_seed"] for j in scene_jobs()}
    scanned, recorded = [], set()
    for directory in sorted(Path(output).parent.iterdir()):
        if not directory.is_dir():
            continue
        for stream in ("episodes", "trajectories"):
            if not (directory / f"{stream}.csv").exists():
                continue
            scanned.append(str(directory / f"{stream}.csv"))
            for row in iter_records(directory, stream):
                value = row.get("episode_seed")
                if value is not None:
                    recorded.add(int(value))
    collisions = sorted(reserved & recorded)
    if collisions:
        raise ValueError(f"Predetermined probe seeds collide with recorded environment seeds: {collisions}")
    return dict(scanned=scanned, recorded_unique_seeds=len(recorded), collisions=[])


class _RecordingPolicy:
    def __init__(self, policy, stop_requested):
        self.policy, self.stop_requested = policy, stop_requested
        self.frames = []

    def reset(self):
        self.policy.reset()
        self.frames = []

    def act(self, state, side, action_ids):
        # The collector checks between episodes, so a stop retains the current
        # complete public trajectory instead of discarding its action prefix.
        actions = self.policy.act(state, side, action_ids)
        if not self.frames or int(self.frames[-1]["state"]["step"]) != int(state.step):
            # DecisionState contains only public physical state and past actions.
            self.frames.append(dict(state=state.to_dict(), action_ids=list(map(int, action_ids)),
                                    actions={str(k): int(v) for k, v in actions.items()}))
        return actions

    def episode_q_statistics(self):
        return self.policy.episode_q_statistics()


def collect_state_bank(output, stop_requested=None, on_progress=None):
    from open_score.algos import load_policy
    from open_score.eval.anchors import BLUE_STRATEGY
    from open_score.rules import register_end_to_end_policy, run_episode
    root = Path(output) / "probe"
    with _lock(root / "collection.lock", stop_requested):
        identities = {method: checkpoint_info(run_directory(output, method, 0) / "final.pt",
                                              method=method, seed=0, env="had")
                      for method in ("regir", "refil")}
        contract = dict(schema=1, configs=[list(c) for c in PROBE_CONFIGS], episodes_per_config=100,
                        behavior_checkpoint_ids={k: v["checkpoint_id"] for k, v in identities.items()},
                        seed_base=200000, steps=100,
                        sampling="4 uniform live times; first 2 alive observers and first 5 alive enemies in stable roster order",
                        splits="ID 50/25/25; OOD 50/50; stratified by behavior; scene-disjoint")
        manifest = root / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != contract:
            raise ValueError("Probe state-bank identity differs from the frozen behavior checkpoints/protocol")
        separation = _verify_seed_separation(output)
        atomic_json(root / "seed_separation.json", separation)
        if not manifest.exists():
            atomic_json(manifest, contract)
        policies = {}
        collection_id = uuid.uuid4().hex
        completed = 0
        for job in scene_jobs():
            _check_stop(stop_requested)
            path = root / "trajectories" / f"{_scene_name(job)}.json.gz"
            if path.exists():
                existing = _read_trajectory(path)
                if existing["job"] != job or existing["checkpoint_id"] != identities[job["behavior"]]["checkpoint_id"]:
                    raise ValueError(f"State-bank trajectory identity mismatch: {path}")
                if len(existing["frames"]) != existing["steps"] or not existing["frames"]:
                    raise ValueError(f"Incomplete trajectory cannot be reused: {path}")
                completed += 1
                continue
            method = job["behavior"]
            if method not in policies:
                policy = load_policy(method, identities[method]["path"])
                policy.set_device("cpu")
                if method == "regir":
                    policy.set_eval_depth(4)
                recording = _RecordingPolicy(policy, stop_requested)
                name = f"main0921_probe_{method}_{collection_id}"
                # The process-wide registry rejects duplicate names. Reuse one
                # recorder per behavior; run_episode resets it before each scene.
                # A new collection ID also permits stop/resume in this process.
                register_end_to_end_policy("red", name, "shared public relationship-probe trajectory",
                                           lambda _seed, wrapped=recording: wrapped)
                policies[method] = recording, name
            recording, name = policies[method]
            red, blue, targets = job["config"]
            result = run_episode(targets=targets, red=red, blue=blue, seed=job["episode_seed"],
                                 red_strategy={"architecture": "end_to_end", "policy": name},
                                 blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                 task_mode="damage", spatial_dim=2, target_initialization="random",
                                 diagnostics=False, record_events=False, retain_trajectory=False)
            if len(recording.frames) != int(result["steps"]):
                raise ValueError("Behavior recording lost a decision/action prefix")
            _write_trajectory(path, dict(job=job, checkpoint_id=identities[method]["checkpoint_id"],
                                         steps=int(result["steps"]), frames=recording.frames))
            completed += 1
            _progress(on_progress, phase="probe_collect", completed=completed, total=500)
        atomic_json(root / "collection_complete.json", dict(trajectories=completed, **contract))
    return completed


def selected_times(frames):
    valid = [i for i, frame in enumerate(frames)
             if any(e["health"] > 0 for e in frame["state"]["red"])
             and any(e["health"] > 0 for e in frame["state"]["blue"])]
    return [] if not valid else [valid[i] for i in np.unique(np.linspace(0, len(valid) - 1, min(4, len(valid))).astype(int))]


def teacher_forced_act(policy, state, actions, action_ids):
    """Advance this model's GRU once, then store the actual behavior action."""
    import torch
    select = policy.mac.select_actions
    def forced(batch, *args, **kwargs):
        predicted = select(batch, *args, **kwargs)
        result = torch.zeros_like(predicted)
        inverse = {int(native): i for i, native in enumerate(action_ids)}
        roster = tuple(e.id for e in state.red) if policy.roster is None else policy.roster["red"]
        for i, identity in enumerate(roster):
            result[0, i] = inverse[int(actions[str(identity)] if str(identity) in actions else actions[identity])]
        return result
    policy.mac.select_actions = forced
    try:
        return policy.act(state, "red", action_ids)
    finally:
        policy.mac.select_actions = select


def relationship_labels(state, observer, enemy, visible):
    """Only entities visible to this observer contribute to any relation."""
    from open_score.envs.features import MAX_AGENTS, MAX_BLUE, POSITION_SCALE
    values = np.full(3, np.nan, dtype=np.float64)
    excluded = Counter()
    reds = [(i, e) for i, e in enumerate(state.red) if e.alive and visible[i] and i != observer]
    blues = [(i, e) for i, e in enumerate(state.blue) if e.alive and visible[MAX_AGENTS + i]]
    targets = [e for i, e in enumerate(state.targets) if visible[MAX_AGENTS + MAX_BLUE + i]]
    if not visible[MAX_AGENTS + enemy] or not state.blue[enemy].alive:
        return values, Counter({f"{label}:enemy_invisible_or_dead": 1 for label in LABELS})
    pos = np.asarray(state.blue[enemy].position[:2])
    if reds:
        values[1] = min(np.linalg.norm(pos - np.asarray(e.position[:2])) for _, e in reds) / POSITION_SCALE
    else:
        excluded[f"{LABELS[1]}:no_other_visible_defender"] += 1
    if not targets:
        for label in (LABELS[0], LABELS[2]):
            excluded[f"{label}:no_visible_target"] += 1
        return values, excluded
    points = np.asarray([e.position[:2] for e in targets])
    def nearest(entity_position):
        distances = np.linalg.norm(points - np.asarray(entity_position[:2]), axis=1)
        minimum = distances.min()
        tied = np.flatnonzero(np.isclose(distances, minimum, rtol=0, atol=1e-9))
        return minimum, int(tied[0]) if len(tied) == 1 else None
    distance, region = nearest(pos)
    values[0] = distance / POSITION_SCALE
    if region is None:
        excluded[f"{LABELS[2]}:geometric_target_tie"] += 1
        return values, excluded
    regions_b = [nearest(e.position)[1] for _, e in blues]
    regions_r = [nearest(e.position)[1] for _, e in reds]
    if None in regions_b + regions_r:
        excluded[f"{LABELS[2]}:geometric_target_tie"] += 1
        return values, excluded
    nb, nr = regions_b.count(region), regions_r.count(region)
    if nb + nr:
        values[2] = (nb - nr) / (nb + nr)
    else:
        excluded[f"{LABELS[2]}:empty_region"] += 1
    return values, excluded


def _random_policy(template, seed):
    import copy
    import torch
    from open_score.models.entity_encoder import EntityAgent
    from open_score.envs.entity_env import FrozenPolicyAdapter
    mac = copy.deepcopy(template.mac)
    args = copy.deepcopy(template.args)
    input_shape = int(args.entity_shape) + (int(args.n_actions) if args.entity_last_action else 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        mac.agent = EntityAgent(input_shape, args)
    mac.args = args
    return FrozenPolicyAdapter(mac, args, template.scheme, template.groups, template.preprocess, mixer=None)


def extract_scene(policy, trajectory, stop_requested=None):
    from open_score.envs import DecisionState
    from open_score.envs.features import MAX_AGENTS
    policy.reset()
    policy.set_eval_depth(4)
    agent = policy.mac.agent
    times = set(selected_times(trajectory["frames"]))
    features, labels, metadata = [], [], []
    captures = []
    exclusions = Counter()
    for index, frame in enumerate(trajectory["frames"]):
        _check_stop(stop_requested)
        state = DecisionState.from_dict(frame["state"])
        if index in times:
            agent.enable_capture(max_decisions=1, max_observers=2)
        teacher_forced_act(policy, state, frame["actions"], frame["action_ids"])
        if index not in times:
            continue
        if len(agent.last_capture) != 1:
            raise ValueError("Real-decision capture missing from teacher-forced policy")
        capture = agent.last_capture[0]
        agent.disable_capture()
        enemies = [i for i, e in enumerate(state.blue) if e.alive][:5]
        captures.append(capture)
        for row, observer in enumerate(capture["observer_ids"].tolist()):
            origins = capture["origin"][row].numpy().astype(int)
            visible = np.zeros(len(origins), dtype=bool)
            visible[origins] = ~capture["key_mask"][row].numpy()
            for enemy in enemies:
                label, excluded = relationship_labels(state, observer, enemy, visible)
                exclusions.update(excluded)
                position = int(np.flatnonzero(origins == MAX_AGENTS + enemy)[0])
                features.append(capture["H"][row, :, position].numpy())
                labels.append(label)
                metadata.append((state.step, observer, enemy))
    agent.disable_capture()
    width = int(policy.args.global_embed_dim)
    return dict(X=np.asarray(features, dtype=np.float32).reshape(-1, 5, width),
                y=np.asarray(labels, dtype=np.float64).reshape(-1, 3),
                pairs=np.asarray(metadata, dtype=np.int32).reshape(-1, 3),
                captures=captures, exclusions=dict(exclusions), selected_times=sorted(times))


def _save_features(path, payload):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(payload, pending)
    os.replace(pending, path)


def r_squared(y, predicted):
    if len(y) < 2:
        return None
    denominator = float(np.square(y - np.mean(y)).sum())
    if denominator <= np.finfo(float).eps:
        return None
    return 1. - float(np.square(y - predicted).sum()) / denominator


def ridge_fit(x, y, alpha):
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z, offset = (x - mean) / scale, float(y.mean())
    weights = np.linalg.solve(z.T @ z + float(alpha) * np.eye(z.shape[1]), z.T @ (y - offset))
    return mean, scale, weights, offset


def ridge_predict(fit, x):
    mean, scale, weights, offset = fit
    return (x - mean) / scale @ weights + offset


def fit_probe(features, model_seed, control):
    """Choose one alpha per label/control across H0..H4 on ID validation only."""
    rows = []
    x = np.concatenate([f["X"] for f in features], axis=0).astype(np.float64)
    y = np.concatenate([f["y"] for f in features], axis=0)
    splits = np.concatenate([np.repeat(f["job"]["split"], len(f["X"])) for f in features])
    configs = np.concatenate([np.repeat(_config_name(f["job"]["config"]), len(f["X"])) for f in features])
    for label_index, label in enumerate(LABELS):
        valid = np.isfinite(y[:, label_index])
        train = (splits == "id_train") & valid
        validation = (splits == "id_val") & valid
        target = y[train, label_index].copy()
        if not len(target) or not validation.any():
            raise ValueError(f"No defined ID train/validation samples for {label}")
        if control == "shuffled_labels":
            target = np.random.default_rng(610000 + 100 * int(model_seed) + label_index).permutation(target)
        scores = []
        for alpha in ALPHAS:
            metrics = [r_squared(y[validation, label_index], ridge_predict(
                ridge_fit(x[train, r], target, alpha), x[validation, r])) for r in range(5)]
            finite = [v for v in metrics if v is not None]
            scores.append(float(np.mean(finite)) if finite else -np.inf)
        alpha = ALPHAS[int(np.argmax(scores))]
        for r in range(5):
            fitted = ridge_fit(x[train, r], target, alpha)
            for split in ("id_train", "id_val", "id_test", "ood_explore", "ood_confirm"):
                for config in sorted(set(configs[splits == split])):
                    selected = valid & (splits == split) & (configs == config)
                    metric = r_squared(y[selected, label_index], ridge_predict(fitted, x[selected, r])) if selected.any() else None
                    rows.append(dict(model_seed=int(model_seed), control=control, label=label, round=r,
                                     split=split, config=config, n=int(selected.sum()), r2=metric, alpha=alpha))
    return rows


def _write_results(root, seed, rows, stop_requested):
    with _lock(root / "results.lock", stop_requested):
        path = root / "results.csv"
        previous = []
        if path.exists():
            with path.open(newline="") as stream:
                previous = [r for r in csv.DictReader(stream) if int(r["model_seed"]) != int(seed)]
        pending = path.with_suffix(".csv.pending")
        with pending.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(previous + rows)
        os.replace(pending, path)


def evaluate_probe(*, output, seed, stop_requested=None, on_progress=None, collect_only=False):
    """One shared bank, then trained/random features and ridge fits for one seed."""
    import torch
    torch.set_num_threads(1)
    from open_score.algos import load_policy
    root = Path(output) / "probe"
    if int(seed) not in (0, 1, 2):
        raise ValueError("Probe model seeds are frozen to 0,1,2")
    try:
        collect_state_bank(output, stop_requested, on_progress)
        if collect_only:
            return dict(status="complete", trajectories=500, collect_only=True)
        with _lock(root / f"seed_{int(seed)}.lock", stop_requested):
            info = checkpoint_info(run_directory(output, "regir", seed) / "final.pt",
                                   method="regir", seed=seed, env="had")
            template = load_policy("regir", info["path"])
            template.set_device("cpu")
            rows, totals = [], {}
            for control in ("trained", "random_init"):
                policy = template if control == "trained" else _random_policy(template, seed)
                policy.set_eval_depth(4)
                features = []
                excluded = Counter()
                for index, job in enumerate(scene_jobs()):
                    _check_stop(stop_requested)
                    path = root / "features" / f"seed_{int(seed)}" / control / f"{_scene_name(job)}.pt"
                    if path.exists():
                        feature = torch.load(path, map_location="cpu", weights_only=False)
                        if feature["checkpoint_id"] != info["checkpoint_id"] or feature["job"] != job:
                            raise ValueError(f"Feature identity mismatch: {path}")
                    else:
                        trajectory = _read_trajectory(root / "trajectories" / f"{_scene_name(job)}.json.gz")
                        feature = dict(extract_scene(policy, trajectory, stop_requested), job=job,
                                       checkpoint_id=info["checkpoint_id"], model_seed=int(seed), control=control)
                        _save_features(path, feature)
                    excluded.update(feature["exclusions"])
                    # Do not retain all H/u snapshots while fitting; they stay in shards.
                    features.append({key: feature[key] for key in ("X", "y", "job")})
                    _progress(on_progress, phase="probe_features", model_seed=int(seed), control=control,
                              completed=index + 1, total=500)
                rows.extend(fit_probe(features, seed, control))
                if control == "trained":
                    rows.extend(fit_probe(features, seed, "shuffled_labels"))
                totals[control] = dict(samples=sum(len(f["X"]) for f in features), exclusions=dict(excluded))
            _write_results(root, seed, rows, stop_requested)
            atomic_json(root / f"seed_{int(seed)}.complete.json", dict(status="complete", model_seed=int(seed),
                        checkpoint_id=info["checkpoint_id"], controls=totals, results_rows=len(rows)))
            return dict(status="complete", model_seed=int(seed), results_rows=len(rows), trajectories=500)
    except ProbeStopped:
        return dict(status="stopped", model_seed=int(seed))
    except TimeoutError as error:
        return dict(status="blocked", model_seed=int(seed), reason=str(error))
