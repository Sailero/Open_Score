"""main0923 frozen-model probes (G6): coverage, dynamics, deep rounds, global, readout, intent."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .experiment import atomic_json, checkpoint_info, read_json, run_directory, test_depth
from .relationship_probe import (ProbeStopped, _check_stop, _lock, _progress,
                                 _random_policy, _read_trajectory, _scene_name,
                                 extract_scene, scene_jobs)


PURSUIT_DISTANCE = 1500.0
PURSUIT_ANGLE = np.deg2rad(30.0)
ALPHAS = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)
DEEP_CONFIGS = ((10, 10, 2), (50, 50, 2))
DEEP_PER_CONFIG = 15
SHIFT_FROM = "10v10_K2"
SHIFT_TO = ("30v30_K2", "50v50_K2", "30v30_K12")


def _cfg(job):
    return "%dv%d_K%d" % tuple(job["config"])


def _numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_bool(mask):
    value = _numpy(mask)
    return value.astype(bool) if value is not None else None


def _complete_path(output, kind, method, seed):
    return Path(output) / "probe" / kind / f"{method}_s{int(seed)}.complete.json"


def probe_complete(output, kind, method, seed, checkpoint_id=None):
    data = read_json(_complete_path(output, kind, method, seed), {})
    if not data or data.get("status") != "complete":
        return False
    return checkpoint_id is None or data.get("checkpoint_id") == checkpoint_id


def _rewrite_csv(path, key, rows, fields, stop_requested):
    path = Path(path)
    with _lock(path.with_suffix(path.suffix + ".lock"), stop_requested):
        previous = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                previous = [row for row in csv.DictReader(stream)
                            if (row.get("method"), int(row.get("seed", -1))) != key]
        pending = path.with_suffix(path.suffix + ".pending")
        path.parent.mkdir(parents=True, exist_ok=True)
        with pending.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(previous + rows)
        pending.replace(path)


def _load_policy(output, method, seed):
    from open_score.algos import load_policy
    info = checkpoint_info(run_directory(output, method, seed) / "final.pt",
                           method=method, seed=seed, env="had")
    policy = load_policy(method, info["path"])
    policy.set_device("cpu")
    depth = test_depth(method)
    if depth is not None:
        policy.set_eval_depth(depth)
    return policy, info


def _iter_scenes(policy, output, stop_requested, on_progress, extras=True):
    root = Path(output) / "probe" / "trajectories"
    jobs = scene_jobs()
    for index, job in enumerate(jobs):
        _check_stop(stop_requested)
        trajectory = _read_trajectory(root / f"{_scene_name(job)}.json.gz")
        feature = extract_scene(policy, trajectory, stop_requested=stop_requested, extras=extras)
        _progress(on_progress, phase="probe_scenes", completed=index + 1, total=len(jobs),
                  method=getattr(policy.args, "method", ""), config=_cfg(job))
        yield job, feature


def _pos2(entity):
    return np.asarray(entity.position[:2], dtype=np.float64)


def _vel2(entity):
    return np.asarray(entity.velocity[:2], dtype=np.float64)


def pursues(red, blue):
    offset = _pos2(blue) - _pos2(red)
    distance = np.linalg.norm(offset)
    speed = np.linalg.norm(_vel2(red))
    if distance >= PURSUIT_DISTANCE or speed <= 1e-9:
        return False
    return float(offset @ _vel2(red)) / (distance * speed) > np.cos(PURSUIT_ANGLE)


def _visible_sets(state, capture, row):
    from open_score.envs.features import MAX_AGENTS, MAX_BLUE
    origin = _numpy(capture["origin"][row]).astype(int)
    visible_slot = ~_as_bool(capture["key_mask"][row])
    reds, blues, targets = [], [], []
    for slot, ok in enumerate(visible_slot):
        if not ok:
            continue
        identity = int(origin[slot]) if slot < len(origin) else slot
        if identity < MAX_AGENTS and identity < len(state.red) and state.red[identity].alive:
            reds.append((slot, identity, state.red[identity]))
        elif MAX_AGENTS <= identity < MAX_AGENTS + MAX_BLUE:
            blue = identity - MAX_AGENTS
            if blue < len(state.blue) and state.blue[blue].alive:
                blues.append((slot, blue, state.blue[blue]))
        else:
            target = identity - MAX_AGENTS - MAX_BLUE
            if 0 <= target < len(state.targets):
                targets.append((slot, target, state.targets[target]))
    return reds, blues, targets


def coverage_label(state, observer, enemy, visible_reds):
    """True if a visible live red other than the observer pursues this blue."""
    blue = state.blue[enemy]
    if not blue.alive:
        return None
    others = [red for _, identity, red in visible_reds if identity != observer]
    if not others:
        return None
    return float(any(pursues(red, blue) for red in others))


def global_labels(state, observer, reds, blues, targets):
    """Per-observer scalars. threat = uncovered incoming attackers (user-locked)."""
    from open_score.envs.features import POSITION_SCALE
    values = dict(threat=np.nan, unmarked_frac=np.nan, centroid_gap=np.nan,
                  nearest_target_distance=np.nan)
    if blues:
        unmarked = sum(1 for _, _, blue in blues
                       if not any(pursues(red, blue) for _, _, red in reds))
        values["unmarked_frac"] = unmarked / len(blues)
        if targets:
            uncovered = 0
            for _, _, blue in blues:
                blue_pos = _pos2(blue)
                own = min(targets, key=lambda item: np.linalg.norm(blue_pos - _pos2(item[2])))
                nearest = np.linalg.norm(blue_pos - _pos2(own[2]))
                red_to_own = min((np.linalg.norm(_pos2(red) - _pos2(own[2])) for _, _, red in reds),
                                 default=np.inf)
                uncovered += float(nearest < red_to_own)
            values["threat"] = uncovered / len(blues)
    if reds and blues:
        red_c = np.mean([_pos2(red) for _, _, red in reds], axis=0)
        blue_c = np.mean([_pos2(blue) for _, _, blue in blues], axis=0)
        values["centroid_gap"] = float(np.linalg.norm(red_c - blue_c)) / POSITION_SCALE
    observer_entity = state.red[observer] if observer < len(state.red) else None
    if observer_entity is not None and targets:
        values["nearest_target_distance"] = min(
            np.linalg.norm(_pos2(observer_entity) - _pos2(target)) for _, _, target in targets
        ) / POSITION_SCALE
    return values


def logistic_fit(x, y, alpha):
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = float(np.clip(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)), -10, 10))
    for _ in range(40):
        logits = np.clip(z @ weights + bias, -20, 20)
        probs = 1. / (1. + np.exp(-logits))
        w = probs * (1. - probs)
        residual = probs - y
        grad_w = z.T @ residual + float(alpha) * weights
        grad_b = float(residual.sum())
        hessian = (z * w[:, None]).T @ z + float(alpha) * np.eye(z.shape[1])
        try:
            step = np.linalg.solve(hessian, grad_w)
        except np.linalg.LinAlgError:
            break
        weights = weights - step
        denom = float(w.sum()) + 1e-9
        bias = bias - grad_b / denom
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return mean, scale, weights, bias


def logistic_predict(fit, x):
    mean, scale, weights, bias = fit
    logits = np.clip(((x - mean) / scale) @ weights + bias, -20, 20)
    return 1. / (1. + np.exp(-logits))


def _auc(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    pos, neg = scores[y >= 0.5], scores[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(neg)
    ranked = np.searchsorted(neg[order], pos, side="right")
    tied = np.searchsorted(neg[order], pos, side="left")
    return float((ranked + tied).sum() / (2.0 * len(pos) * len(neg)))


def _accuracy(y, scores):
    return float(((scores >= 0.5) == (y >= 0.5)).mean()) if len(y) else None


def ridge_fit(x, y, alpha):
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z, offset = (x - mean) / scale, float(y.mean())
    eye = np.eye(z.shape[1])
    weights = np.linalg.solve(z.T @ z + float(alpha) * eye, z.T @ (y - offset))
    return mean, scale, weights, offset


def ridge_predict(fit, x):
    mean, scale, weights, offset = fit
    return (x - mean) / scale @ weights + offset


def r_squared(y, predicted):
    if len(y) < 2:
        return None
    denom = float(np.square(y - np.mean(y)).sum())
    if denom <= np.finfo(float).eps:
        return None
    return 1. - float(np.square(y - predicted).sum()) / denom


def _run_or_reuse(output, kind, method, seed, stop_requested, on_progress, fit):
    import torch
    torch.set_num_threads(1)
    root = Path(output) / "probe"
    try:
        policy, info = _load_policy(output, method, seed)
        marker = _complete_path(output, kind, method, seed)
        with _lock(marker.with_suffix(".lock"), stop_requested):
            if probe_complete(output, kind, method, seed, info["checkpoint_id"]):
                return dict(status="complete", reused=True, method=method, seed=int(seed))
            rows = fit(policy, info, stop_requested, on_progress)
            marker.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(marker, dict(status="complete", method=method, seed=int(seed),
                                     checkpoint_id=info["checkpoint_id"], rows=len(rows)))
            return dict(status="complete", method=method, seed=int(seed), rows=len(rows))
    except ProbeStopped:
        return dict(status="stopped", method=method, seed=int(seed))
    except TimeoutError as error:
        return dict(status="blocked", method=method, seed=int(seed), reason=str(error))


def evaluate_coverage_probe(*, output, method, seed, stop_requested=None, on_progress=None):
    fields = ("method", "seed", "control", "round", "split", "config", "n", "auc", "acc", "alpha")

    def fit(policy, info, stop, progress):
        rows = []
        for control, model in (("trained", policy),
                               ("random_init", _random_policy(policy, seed))):
            samples = []
            for job, feature in _iter_scenes(model, output, stop, progress):
                for capture in feature["captures"]:
                    state = capture["state"]
                    for row, observer in enumerate(_numpy(capture["observer_ids"]).tolist()):
                        reds, blues, _ = _visible_sets(state, capture, row)
                        h = _numpy(capture["H"][row])
                        for slot, enemy, _blue in blues:
                            label = coverage_label(state, int(observer), enemy, reds)
                            if label is None:
                                continue
                            samples.append(dict(x=h[:, slot], y=label, split=job["split"],
                                                config=_cfg(job)))
            if not samples:
                continue
            rounds = samples[0]["x"].shape[0]
            x = np.stack([s["x"] for s in samples]).astype(np.float64)
            y = np.asarray([s["y"] for s in samples], dtype=np.float64)
            splits = np.asarray([s["split"] for s in samples])
            configs = np.asarray([s["config"] for s in samples])
            train = splits == "id_train"
            validation = splits == "id_val"
            if not train.any() or not validation.any():
                raise ValueError(f"coverage probe missing ID splits for {method} seed {seed}")
            target = y[train].copy()
            if control == "shuffled_labels":
                continue
            for r in range(rounds):
                scores = []
                for alpha in ALPHAS:
                    fitted = logistic_fit(x[train, r], target, alpha)
                    pred = logistic_predict(fitted, x[validation, r])
                    scores.append(_auc(y[validation], pred) or -np.inf)
                alpha = ALPHAS[int(np.argmax(scores))]
                fitted = logistic_fit(x[train, r], target, alpha)
                for split in ("id_train", "id_val", "id_test", "ood_explore", "ood_confirm"):
                    for config in sorted(set(configs[splits == split])):
                        selected = (splits == split) & (configs == config)
                        if not selected.any():
                            continue
                        pred = logistic_predict(fitted, x[selected, r])
                        rows.append(dict(method=method, seed=int(seed), control=control, round=r,
                                         split=split, config=config, n=int(selected.sum()),
                                         auc=_auc(y[selected], pred), acc=_accuracy(y[selected], pred),
                                         alpha=alpha))
            if control == "trained":
                rng = np.random.default_rng(710000 + 100 * int(seed))
                shuffled = y[train].copy()
                rng.shuffle(shuffled)
                for r in range(rounds):
                    scores = []
                    for alpha in ALPHAS:
                        fitted = logistic_fit(x[train, r], shuffled, alpha)
                        pred = logistic_predict(fitted, x[validation, r])
                        scores.append(_auc(y[validation], pred) or -np.inf)
                    alpha = ALPHAS[int(np.argmax(scores))]
                    fitted = logistic_fit(x[train, r], shuffled, alpha)
                    for split in ("id_train", "id_val", "id_test", "ood_explore", "ood_confirm"):
                        for config in sorted(set(configs[splits == split])):
                            selected = (splits == split) & (configs == config)
                            if not selected.any():
                                continue
                            pred = logistic_predict(fitted, x[selected, r])
                            rows.append(dict(method=method, seed=int(seed), control="shuffled_labels",
                                             round=r, split=split, config=config, n=int(selected.sum()),
                                             auc=_auc(y[selected], pred), acc=_accuracy(y[selected], pred),
                                             alpha=alpha))
        _rewrite_csv(Path(output) / "probe" / "coverage.csv", (method, int(seed)), rows, fields, stop)
        return rows

    return _run_or_reuse(output, "coverage", method, seed, stop_requested, on_progress, fit)


def evaluate_dynamics(*, output, method, seed, stop_requested=None, on_progress=None):
    import torch.nn.functional as F
    fields = ("method", "seed", "config", "round", "n", "alpha", "cosine", "spread",
              "readout_norm", "xcos_fused", "xcos_round", "token_n", "mean_norm", "sum_sq",
              "shift_from_10v10")

    def fit(policy, info, stop, progress):
        stats = {}
        width = None
        for job, feature in _iter_scenes(policy, output, stop, progress):
            cfg = _cfg(job)
            for capture in feature["captures"]:
                h, mask = _numpy(capture["H"]), _as_bool(capture["key_mask"])
                alpha = _numpy(capture.get("alpha"))
                read = _numpy(capture.get("u"))
                if h is None:
                    continue
                width = h.shape[-1]
                acc = stats.setdefault(cfg, dict(n=0, npair=0, xcos_fused=0.0,
                                                 alpha=None, cos=None, spread=None, unorm=None,
                                                 xcos=None, token_n=None, token_sum=None, token_sq=None))
                rounds = h.shape[1]
                if acc["cos"] is None:
                    acc.update(alpha=np.zeros(max(rounds - 1, 0)), cos=np.zeros(rounds),
                               spread=np.zeros(rounds), unorm=np.zeros(max(rounds - 1, 0)),
                               xcos=np.zeros(max(rounds - 1, 0)), token_n=np.zeros(rounds),
                               token_sum=np.zeros((rounds, width)), token_sq=np.zeros(rounds))
                for i in range(h.shape[0]):
                    valid = mask[i] if mask is not None else np.ones(h.shape[2], dtype=bool)
                    valid = ~valid if mask is not None else valid
                    if int(valid.sum()) < 2:
                        continue
                    acc["n"] += 1
                    for r in range(rounds):
                        tokens = h[i, r, valid]
                        normed = tokens / np.maximum(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-8)
                        m = tokens.shape[0]
                        acc["cos"][r] += float(((normed @ normed.T).sum() - m) / (m * (m - 1)))
                        acc["spread"][r] += float(np.linalg.norm(tokens - tokens.mean(0), axis=-1).mean()
                                                  / max(float(np.linalg.norm(tokens, axis=-1).mean()), 1e-8))
                        acc["token_n"][r] += m
                        acc["token_sum"][r] += tokens.sum(0)
                        acc["token_sq"][r] += float((tokens ** 2).sum())
                    if alpha is not None:
                        acc["alpha"][:len(alpha[i])] += alpha[i]
                    if read is not None:
                        acc["unorm"][:read.shape[1]] += np.linalg.norm(read[i], axis=-1)
                if h.shape[0] == 2 and read is not None and alpha is not None:
                    fused = (alpha[..., None] * read).sum(1)
                    acc["npair"] += 1
                    acc["xcos_fused"] += float(F.cosine_similarity(
                        __import__("torch").as_tensor(fused[0]),
                        __import__("torch").as_tensor(fused[1]), dim=0))
                    for r in range(read.shape[1]):
                        acc["xcos"][r] += float(F.cosine_similarity(
                            __import__("torch").as_tensor(read[0, r]),
                            __import__("torch").as_tensor(read[1, r]), dim=0))
        means = {}
        for cfg, acc in stats.items():
            n = max(acc["n"], 1)
            means[cfg] = acc
            acc["alpha"] = acc["alpha"] / n
            acc["cos"] = acc["cos"] / n
            acc["spread"] = acc["spread"] / n
            acc["unorm"] = acc["unorm"] / n
            acc["xcos"] = acc["xcos"] / max(acc["npair"], 1)
            acc["xcos_fused"] = acc["xcos_fused"] / max(acc["npair"], 1)
        rows = []
        for cfg, acc in sorted(means.items()):
            rounds = len(acc["cos"])
            for r in range(rounds):
                shift = ""
                if cfg != SHIFT_FROM and SHIFT_FROM in means and acc["token_n"][r] and means[SHIFT_FROM]["token_n"][r]:
                    base, other = means[SHIFT_FROM], acc
                    mu0 = base["token_sum"][r] / base["token_n"][r]
                    mu1 = other["token_sum"][r] / other["token_n"][r]
                    v0 = base["token_sq"][r] / base["token_n"][r] - float((mu0 ** 2).sum())
                    v1 = other["token_sq"][r] / other["token_n"][r] - float((mu1 ** 2).sum())
                    shift = float(((mu0 - mu1) ** 2).sum()) / max((v0 + v1) / 2, 1e-12)
                rows.append(dict(method=method, seed=int(seed), config=cfg, round=r, n=int(acc["n"]),
                                 alpha=float(acc["alpha"][r - 1]) if r else "",
                                 cosine=float(acc["cos"][r]), spread=float(acc["spread"][r]),
                                 readout_norm=float(acc["unorm"][r - 1]) if r else "",
                                 xcos_fused=float(acc["xcos_fused"]) if r == 0 else "",
                                 xcos_round=float(acc["xcos"][r - 1]) if r else "",
                                 token_n=int(acc["token_n"][r]),
                                 mean_norm=float(np.linalg.norm(acc["token_sum"][r] / max(acc["token_n"][r], 1))),
                                 sum_sq=float(acc["token_sq"][r]), shift_from_10v10=shift))
        _rewrite_csv(Path(output) / "probe" / "dynamics.csv", (method, int(seed)), rows, fields, stop)
        return rows

    return _run_or_reuse(output, "dynamics", method, seed, stop_requested, on_progress, fit)


def evaluate_deep_rounds(*, output, method, seed, stop_requested=None, on_progress=None, max_depth=8):
    import torch
    import torch.nn.functional as F
    fields = ("method", "seed", "config", "round", "n", "norm", "ratio", "cosine")

    def fit(policy, info, stop, progress):
        net = getattr(policy.mac.agent, "global_net", None)
        if net is None or getattr(policy.args, "rer_update", "tied") == "untied4":
            rows = []
            _rewrite_csv(Path(output) / "probe" / "deep_rounds.csv", (method, int(seed)), rows, fields, stop)
            return rows
        picked, seen = [], Counter()
        for job in scene_jobs():
            key = tuple(job["config"])
            if key in DEEP_CONFIGS and seen[key] < DEEP_PER_CONFIG:
                picked.append(job)
                seen[key] += 1
        acc = {}
        with torch.inference_mode():
            for index, job in enumerate(picked):
                _check_stop(stop)
                trajectory = _read_trajectory(Path(output) / "probe" / "trajectories"
                                              / f"{_scene_name(job)}.json.gz")
                feature = extract_scene(policy, trajectory, stop_requested=stop, extras=True)
                _progress(progress, phase="deep_rounds", completed=index + 1, total=len(picked))
                cfg = _cfg(job)
                for capture in feature["captures"]:
                    h0 = capture["H"][:, 0]
                    if hasattr(h0, "numpy"):
                        h0_t = capture["H"][:, 0]
                    else:
                        h0_t = torch.as_tensor(h0)
                    km = capture["key_mask"] if torch.is_tensor(capture["key_mask"]) else torch.as_tensor(capture["key_mask"])
                    types = capture["types"] if torch.is_tensor(capture["types"]) else torch.as_tensor(capture["types"])
                    origin = capture["origin"] if torch.is_tensor(capture["origin"]) else torch.as_tensor(capture["origin"])
                    depth = torch.full((h0_t.shape[0],), int(max_depth), dtype=torch.long)
                    extra = {}
                    if getattr(net, "intent", False):
                        extra["origin"] = origin
                    states, _ = net.build_memories(h0_t, km, types, depth, **extra)
                    bucket = acc.setdefault(cfg, dict(n=0, norm=[0.0] * (max_depth + 1),
                                                      cos=[0.0] * (max_depth + 1)))
                    mask = (~km).numpy() if torch.is_tensor(km) else ~np.asarray(km, dtype=bool)
                    for i in range(h0_t.shape[0]):
                        valid = mask[i]
                        if int(valid.sum()) < 2:
                            continue
                        bucket["n"] += 1
                        sequence = [h0_t] + list(states)
                        for r, memory in enumerate(sequence):
                            tokens = memory[i][valid]
                            bucket["norm"][r] += float(tokens.norm(dim=-1).mean())
                            normed = F.normalize(tokens, dim=-1)
                            m = tokens.shape[0]
                            bucket["cos"][r] += float(((normed @ normed.T).sum() - m) / (m * (m - 1)))
        rows = []
        for cfg, bucket in sorted(acc.items()):
            n = max(bucket["n"], 1)
            norms = [value / n for value in bucket["norm"]]
            cos = [value / n for value in bucket["cos"]]
            for r, norm in enumerate(norms):
                ratio = norms[r] / norms[r - 1] if r and norms[r - 1] else ""
                rows.append(dict(method=method, seed=int(seed), config=cfg, round=r, n=int(bucket["n"]),
                                 norm=norm, ratio=ratio, cosine=cos[r]))
        _rewrite_csv(Path(output) / "probe" / "deep_rounds.csv", (method, int(seed)), rows, fields, stop)
        return rows

    return _run_or_reuse(output, "deep_rounds", method, seed, stop_requested, on_progress, fit)


def evaluate_global_probe(*, output, method, seed, stop_requested=None, on_progress=None):
    labels = ("threat", "unmarked_frac", "centroid_gap", "nearest_target_distance")
    fields = ("method", "seed", "control", "label", "feature", "round", "split", "config",
              "n", "r2", "alpha")

    def collect(model, stop, progress):
        samples = []
        for job, feature in _iter_scenes(model, output, stop, progress):
            for capture in feature["captures"]:
                state = capture["state"]
                for row, observer in enumerate(_numpy(capture["observer_ids"]).tolist()):
                    reds, blues, targets = _visible_sets(state, capture, row)
                    values = global_labels(state, int(observer), reds, blues, targets)
                    h = _numpy(capture["H"][row])
                    mask = ~_as_bool(capture["key_mask"][row])
                    pooled = h[:, mask].mean(1) if mask.any() else np.zeros((h.shape[0], h.shape[-1]))
                    read = _numpy(capture.get("u"))
                    read = None if read is None else read[row]
                    samples.append(dict(values=values, pooled=pooled, read=read,
                                        split=job["split"], config=_cfg(job)))
        return samples

    def fit_feature(samples, control, feature_name, getter, stop):
        rows = []
        first = next((getter(s) for s in samples if getter(s) is not None), None)
        if first is None:
            return rows
        rounds = first.shape[0]
        splits = np.asarray([s["split"] for s in samples])
        configs = np.asarray([s["config"] for s in samples])
        for label in labels:
            y = np.asarray([s["values"][label] for s in samples], dtype=np.float64)
            valid = np.isfinite(y)
            train = (splits == "id_train") & valid
            validation = (splits == "id_val") & valid
            if not train.any() or not validation.any():
                continue
            target = y[train].copy()
            if control == "shuffled_labels":
                target = np.random.default_rng(720000 + 100 * int(seed) + labels.index(label)).permutation(target)
            x = np.stack([getter(s) for s in samples]).astype(np.float64)
            scores = []
            for alpha in ALPHAS:
                metrics = [r_squared(y[validation], ridge_predict(ridge_fit(x[train, r], target, alpha),
                                                                  x[validation, r]))
                           for r in range(rounds)]
                finite = [v for v in metrics if v is not None]
                scores.append(float(np.mean(finite)) if finite else -np.inf)
            alpha = ALPHAS[int(np.argmax(scores))]
            for r in range(rounds):
                fitted = ridge_fit(x[train, r], target, alpha)
                for split in ("id_train", "id_val", "id_test", "ood_explore", "ood_confirm"):
                    for config in sorted(set(configs[splits == split])):
                        selected = valid & (splits == split) & (configs == config)
                        if not selected.any():
                            continue
                        metric = r_squared(y[selected], ridge_predict(fitted, x[selected, r]))
                        rows.append(dict(method=method, seed=int(seed), control=control, label=label,
                                         feature=feature_name, round=r, split=split, config=config,
                                         n=int(selected.sum()), r2=metric, alpha=alpha))
        return rows

    def fit(policy, info, stop, progress):
        rows = []
        trained = collect(policy, stop, progress)
        random = collect(_random_policy(policy, seed), stop, progress)
        for control, samples in (("trained", trained), ("random_init", random),
                                 ("shuffled_labels", trained)):
            rows += fit_feature(samples, control, "h_mean", lambda s: s["pooled"], stop)
            rows += fit_feature(samples, control, "readout", lambda s: s["read"], stop)
        _rewrite_csv(Path(output) / "probe" / "global_results.csv", (method, int(seed)), rows, fields, stop)
        return rows

    return _run_or_reuse(output, "global_probe", method, seed, stop_requested, on_progress, fit)


def evaluate_readout_attention(*, output, method, seed, stop_requested=None, on_progress=None):
    fields = ("method", "seed", "config", "round", "n", "share_pursued", "share_unmarked",
              "share_teammate", "share_target", "share_other")

    def fit(policy, info, stop, progress):
        acc = {}
        for job, feature in _iter_scenes(policy, output, stop, progress):
            cfg = _cfg(job)
            for capture in feature["captures"]:
                weights = _numpy(capture.get("read_attn"))
                if weights is None:
                    continue
                state = capture["state"]
                for row, observer in enumerate(_numpy(capture["observer_ids"]).tolist()):
                    reds, blues, targets = _visible_sets(state, capture, row)
                    pursued = {slot for slot, _, blue in blues
                               if any(pursues(red, blue) for _, identity, red in reds
                                      if identity != observer)}
                    unmarked = {slot for slot, _, blue in blues if slot not in pursued}
                    teammates = {slot for slot, identity, _ in reds if identity != observer}
                    target_slots = {slot for slot, _, _ in targets}
                    w = weights[row]
                    if w.ndim == 1:
                        w = w[None]
                    for r in range(w.shape[0]):
                        mass = float(w[r].sum())
                        if mass <= 1e-12:
                            continue
                        bucket = acc.setdefault((cfg, r), dict(n=0, pursued=0, unmarked=0,
                                                               teammate=0, target=0, other=0))
                        bucket["n"] += 1
                        bucket["pursued"] += float(sum(w[r, s] for s in pursued)) / mass
                        bucket["unmarked"] += float(sum(w[r, s] for s in unmarked)) / mass
                        bucket["teammate"] += float(sum(w[r, s] for s in teammates)) / mass
                        bucket["target"] += float(sum(w[r, s] for s in target_slots)) / mass
                        used = pursued | unmarked | teammates | target_slots
                        bucket["other"] += float(sum(w[r, s] for s in range(w.shape[1])
                                                     if s not in used)) / mass
        rows = [dict(method=method, seed=int(seed), config=cfg, round=r, n=int(b["n"]),
                     share_pursued=b["pursued"] / b["n"], share_unmarked=b["unmarked"] / b["n"],
                     share_teammate=b["teammate"] / b["n"], share_target=b["target"] / b["n"],
                     share_other=b["other"] / b["n"])
                for (cfg, r), b in sorted(acc.items())]
        _rewrite_csv(Path(output) / "probe" / "readout_attention.csv", (method, int(seed)), rows, fields, stop)
        return rows

    return _run_or_reuse(output, "readout_attention", method, seed, stop_requested, on_progress, fit)


def evaluate_intent_accuracy(*, output, method, seed, stop_requested=None, on_progress=None):
    import torch
    import torch.nn.functional as F
    fields = ("method", "seed", "control", "config", "round", "n", "acc", "nll")

    def _labels(capture):
        record = capture.get("intent")
        if not record:
            return None
        state, actions, action_ids = capture["state"], capture["actions"], capture["action_ids"]
        inverse = {int(native): i for i, native in enumerate(action_ids)}
        origin = record["origin"]
        teammate = record["teammate"]
        if not torch.is_tensor(origin):
            origin = torch.as_tensor(origin)
        if not torch.is_tensor(teammate):
            teammate = torch.as_tensor(teammate)
        ids = [entity.id for entity in state.red]
        labels = torch.full(origin.shape, -1, dtype=torch.long)
        for i in range(origin.shape[0]):
            for j in range(origin.shape[1]):
                if not bool(teammate[i, j]):
                    continue
                idx = int(origin[i, j])
                if idx < 0 or idx >= len(ids):
                    continue
                native = actions.get(str(ids[idx]), actions.get(ids[idx]))
                if native is None or int(native) not in inverse:
                    continue
                labels[i, j] = inverse[int(native)]
        return labels, teammate

    def collect(model, stop, progress):
        rows = []
        majority = Counter()
        for job, feature in _iter_scenes(model, output, stop, progress):
            for capture in feature["captures"]:
                packed = _labels(capture)
                if packed is None:
                    continue
                labels, teammate = packed
                for round_id, logits in enumerate(capture["intent"]["logits"]):
                    if not torch.is_tensor(logits):
                        logits = torch.as_tensor(logits)
                    keep = teammate & (labels >= 0)
                    if hasattr(capture["intent"].get("depth"), "shape"):
                        depth = capture["intent"]["depth"]
                        keep = keep & (depth > round_id).unsqueeze(-1)
                    n = int(keep.sum())
                    if n == 0:
                        continue
                    chosen, truth = logits[keep], labels[keep]
                    majority.update(int(v) for v in truth.tolist())
                    rows.append(dict(config=_cfg(job), round=round_id + 1, n=n,
                                     acc=float((chosen.argmax(-1) == truth).float().mean()),
                                     nll=float(F.cross_entropy(chosen, truth)),
                                     truth=truth.detach().cpu().numpy(),
                                     logits=chosen.detach().cpu().numpy()))
        return rows, majority

    def fit(policy, info, stop, progress):
        trained, majority = collect(policy, stop, progress)
        random_rows, _ = collect(_random_policy(policy, seed), stop, progress)
        common = majority.most_common(1)[0][0] if majority else 0
        written = []
        for control, items in (("trained", trained), ("random_init", random_rows),
                               ("majority", trained)):
            grouped = {}
            for item in items:
                key = (item["config"], item["round"])
                grouped.setdefault(key, []).append(item)
            for (config, round_id), bundle in sorted(grouped.items()):
                n = sum(item["n"] for item in bundle)
                if control == "majority":
                    hits = sum(int((item["truth"] == common).sum()) for item in bundle)
                    nll = float(-np.log(max(majority[common] / max(sum(majority.values()), 1), 1e-9)))
                    acc = hits / max(n, 1)
                else:
                    acc = sum(item["acc"] * item["n"] for item in bundle) / max(n, 1)
                    nll = sum(item["nll"] * item["n"] for item in bundle) / max(n, 1)
                written.append(dict(method=method, seed=int(seed), control=control, config=config,
                                    round=round_id, n=n, acc=acc, nll=nll))
        _rewrite_csv(Path(output) / "probe" / "intent.csv", (method, int(seed)), written, fields, stop)
        return written

    return _run_or_reuse(output, "intent_accuracy", method, seed, stop_requested, on_progress, fit)


DISPATCH = {
    "coverage": evaluate_coverage_probe,
    "dynamics": evaluate_dynamics,
    "deep_rounds": evaluate_deep_rounds,
    "global_probe": evaluate_global_probe,
    "readout_attention": evaluate_readout_attention,
    "intent_accuracy": evaluate_intent_accuracy,
}


def evaluate_diagnostic(*, output, kind, method, seed, stop_requested=None, on_progress=None):
    if kind not in DISPATCH:
        raise ValueError(f"Unknown diagnostic kind {kind!r}")
    return DISPATCH[kind](output=output, method=method, seed=seed,
                          stop_requested=stop_requested, on_progress=on_progress)
