"""Frozen validation and final-evaluation quotas; no environment construction."""
from __future__ import annotations

import math
from itertools import count
from pathlib import Path
import time

from ..envs.scales import SCALE_POOLS, TEST_POOL


def _safe_refresh(output, run=None):
    try:
        import importlib
        from . import report as report_mod
        importlib.reload(report_mod)
        report_mod.refresh_report(output, run=run)
    except Exception as error:
        print(f"report refresh skipped: {type(error).__name__}: {error}", flush=True)

# Checkpoint selection may only ever see these; they are inside the training pool.
VALIDATION_CONFIGS = ((4, 4, 2), (6, 6, 2), (8, 8, 2), (10, 10, 3))
EQUAL_SCALE_CONFIGS = tuple((s.N_R, s.N_B, s.K) for s in SCALE_POOLS["extrapolation_agents"])
RATIO_CONFIGS = tuple((s.N_R, s.N_B, s.K) for s in SCALE_POOLS["extrapolation_ratio"])
TARGET_K_VALUES = (2, 4, 6, 9, 12)
TARGET_SCALE_NS = (10, 15, 20, 30)
TARGET_PLOT_NS = (10, 30)
# scales.py is the source of truth. 1:2 stays named for legacy CSV rows only.
TEST_CONFIGS = tuple((scale.N_R, scale.N_B, scale.K) for scale in TEST_POOL)
# Pool-in plus equal-scale and target axes. 10v10 K2 sits on the equal-scale
# line; 10v10 K3 stays in the validation set.
FINAL_CONFIGS = tuple(dict.fromkeys(VALIDATION_CONFIGS + TEST_CONFIGS))
VALIDATION_EPISODES_PER_CONFIG = 25
FINAL_EPISODES_PER_CONFIG = 300
FINAL_EPISODES = len(FINAL_CONFIGS) * FINAL_EPISODES_PER_CONFIG
# Ordered fixed-slot baselines are sized to the training pool, so they have no
# parameters for a larger roster and are reported on the pool axis only.
POOL_ONLY_METHODS = ("b0_qmix",)
# Shared SelfAttn cycle / slot / feedback. Cardinality attention has no R.
CYCLE_SERIES_METHODS = ("regir", "refil_cycle", "refil_slot", "refil_feedback")
MECH_CONFIG = (30, 30, 2)
MECH_EPISODE_SEEDS = (9000, 9001)
TIMING_METHODS = ("regir", "refil", "refil_matched", "dcg")
TIMING_SCALES = ((10, 10, 2), (50, 50, 2))
DEPTH_SWEEP_DEPTHS = (1, 2, 3, 4, 5, 6)
# Formal frozen eval scores last-iterate weights. Validation-best stays on disk
# as best.pt but does not enter the official tables.
OFFICIAL_CHECKPOINT = "final"
_POLICY_IDS = count()


def final_configs(method=None):
    return VALIDATION_CONFIGS if method in POOL_ONLY_METHODS else FINAL_CONFIGS


def final_episodes(method=None):
    return len(final_configs(method)) * FINAL_EPISODES_PER_CONFIG


def config_dict(config):
    if isinstance(config, dict):
        return {key: int(config[key]) for key in ("N_R", "N_B", "K")}
    red, blue, targets = config
    return {"N_R": int(red), "N_B": int(blue), "K": int(targets)}


def config_key(config):
    value = config_dict(config)
    return value["N_R"], value["N_B"], value["K"]


def config_label(config):
    red, blue, targets = config_key(config)
    return f"{red}v{blue} K{targets}"


def evaluation_thresholds(budget):
    budget = int(budget)
    if budget < 50:
        raise ValueError("A formal 50-point budget must be at least 50 physical steps")
    return [math.ceil(budget * point / 50) for point in range(1, 51)]


def validation_jobs(eval_point, t_env=0):
    if not 1 <= int(eval_point) <= 50:
        raise ValueError("eval_point must be 1..50")
    return [dict(config=config_dict(config), episode_seed=9500 + 25 * index + offset,
                 phase="train_eval", eval_point=int(eval_point), t_env=int(t_env),
                 retain_trajectory=False)
            for index, config in enumerate(VALIDATION_CONFIGS) for offset in range(25)]


def checkpoint_tag(stem, t_env):
    """Identify the retained weights, not just the reusable file name.

    ``best.pt`` / ``final.pt`` are overwritten, so a bare stem cannot
    distinguish two different models and would let an earlier evaluation
    satisfy the quota of a later one.
    """
    return f"{stem}@{int(t_env)}"


def official_checkpoint_tag(t_env, depth=None):
    tag = checkpoint_tag(OFFICIAL_CHECKPOINT, t_env)
    if depth is None:
        return tag
    return f"{tag}/R{int(depth)}"


def parse_checkpoint_stem(checkpoint):
    text = str(checkpoint or "")
    if "@" not in text:
        return None
    return text.partition("@")[0]


def final_jobs(t_env=0, checkpoint=None, method=None):
    checkpoint = OFFICIAL_CHECKPOINT if checkpoint is None else checkpoint
    tag = checkpoint_tag(checkpoint, t_env)
    return [dict(config=config_dict(config), episode_seed=9000 + offset, phase="final_eval",
                 eval_point=50, t_env=int(t_env), checkpoint=tag,
                 retain_trajectory=offset < 2)
            for config in final_configs(method) for offset in range(FINAL_EPISODES_PER_CONFIG)]


def validation_score(rows):
    """Return average D only for one complete, correctly seeded validation point."""
    selected = [row for row in rows if row.get("phase") == "train_eval"]
    if not selected:
        return None
    # Hard barrier against test-set leakage into checkpoint selection.
    outside = {config_key(row["config"]) for row in selected} - set(VALIDATION_CONFIGS)
    if outside:
        raise ValueError(f"Checkpoint selection saw non-training configurations: {sorted(outside)}")
    for field in ("run", "method", "seed", "eval_point"):
        identities = {row[field] for row in selected if row.get(field) is not None}
        if len(identities) > 1:
            raise ValueError("Checkpoint selection requires exactly one run/method/seed/eval_point")
    expected = {(config_key(job["config"]), job["episode_seed"]) for job in validation_jobs(selected[0]["eval_point"])}
    by_episode = {(config_key(row["config"]), int(row["episode_seed"])): row for row in selected}
    if set(by_episode) != expected:
        return None
    values = [float(row["D"]) for row in by_episode.values()]
    if not all(math.isfinite(value) for value in values):
        return None
    return sum(values) / 100


def remaining_jobs(jobs, recorded_rows):
    done = {(row.get("phase"), row.get("eval_point"), config_key(row["config"]),
             int(row["episode_seed"]), row.get("checkpoint")) for row in recorded_rows}
    return [job for job in jobs if (job.get("phase"), job.get("eval_point"), config_key(job["config"]),
            int(job["episode_seed"]), job.get("checkpoint")) not in done]


def is_cycle_series(method):
    return method in CYCLE_SERIES_METHODS


def depth_checkpoint_tag(stem, t_env, depth):
    return f"{checkpoint_tag(stem, t_env)}/R{int(depth)}"


def parse_sweep_depth(checkpoint):
    text = str(checkpoint or "")
    marker = "/R"
    if marker not in text:
        return None
    try:
        return int(text.rsplit(marker, 1)[-1])
    except ValueError:
        return None


def parse_checkpoint_t_env(checkpoint):
    """`best@520055` or `best@1000000/R4` → the t_env baked into the tag."""
    text = str(checkpoint or "")
    if "@" not in text:
        return None
    stem, _, rest = text.partition("@")
    if stem not in ("best", "final"):
        return None
    try:
        return int(rest.split("/")[0])
    except ValueError:
        return None


def run_directory(output, method, seed, run=None):
    from open_score.utils.logging import FORMAL_RUN
    run = FORMAL_RUN if run is None else run
    return Path(output) / method / str(run) / f"seed_{int(seed)}"


def official_eval_path(directory, progress_row=None):
    """Only a completed last iterate is eligible; best is never a substitute."""
    directory = Path(directory)
    final = directory / f"{OFFICIAL_CHECKPOINT}.pt"
    if training_finished(directory, progress_row):
        return final
    return None


def official_eval_t_env(directory, progress_row=None):
    """t_env of the last-iterate weights. Official eval scores ``final.pt``."""
    path = official_eval_path(directory, progress_row)
    if path is None:
        return None
    try:
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        return int(saved["progress"]["t_env"])
    except Exception:
        return None


def official_best_t_env(directory):
    return official_eval_t_env(directory)


def training_finished(directory, progress_row=None):
    path = Path(directory) / "final.pt"
    if not path.exists():
        return False
    try:
        from open_score.algos import setup_runtime
        setup_runtime()
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        return (saved["progress"].get("status") in ("completed", "complete")
                and int(saved["progress"]["t_env"]) >= int(saved["config"]["t_max"]))
    except (OSError, ValueError, KeyError, EOFError):
        return False


def rows_for_official_checkpoint(rows, t_env, phase=None):
    if t_env is None:
        return []
    t_env = int(t_env)
    kept = []
    for row in rows:
        if phase is not None and row.get("phase") != phase:
            continue
        if parse_checkpoint_stem(row.get("checkpoint")) != OFFICIAL_CHECKPOINT:
            continue
        if parse_checkpoint_t_env(row.get("checkpoint")) != t_env:
            continue
        kept.append(row)
    return kept


def _formal_eval_allowed(directory, progress_row, t_env, run):
    from open_score.utils.logging import FORMAL_RUN
    if run != FORMAL_RUN:
        return True
    if not training_finished(directory, progress_row):
        return False
    official = official_eval_t_env(directory, progress_row)
    return official is None or int(official) == int(t_env)


def depth_eval_total():
    return len(EQUAL_SCALE_CONFIGS) * FINAL_EPISODES_PER_CONFIG * len(DEPTH_SWEEP_DEPTHS)


def depth_jobs(t_env=0, checkpoint=None, depth=4):
    checkpoint = OFFICIAL_CHECKPOINT if checkpoint is None else checkpoint
    tag = depth_checkpoint_tag(checkpoint, t_env, depth)
    return [dict(config=config_dict(config), episode_seed=9000 + offset, phase="depth_eval",
                 eval_point=50, t_env=int(t_env), checkpoint=tag, cycle_depth=int(depth),
                 retain_trajectory=False)
            for config in EQUAL_SCALE_CONFIGS for offset in range(FINAL_EPISODES_PER_CONFIG)]


def _reuse_final_as_depth(recorded_rows, t_env, checkpoint=None):
    """Formal 1:1 final_eval is already greedy R=4; do not rerun those episodes."""
    checkpoint = OFFICIAL_CHECKPOINT if checkpoint is None else checkpoint
    tag = checkpoint_tag(checkpoint, t_env)
    equal = set(EQUAL_SCALE_CONFIGS)
    reused = []
    for row in recorded_rows:
        if row.get("phase") != "final_eval" or row.get("checkpoint") != tag:
            continue
        if config_key(row["config"]) not in equal:
            continue
        reused.append(dict(row, phase="depth_eval",
                           checkpoint=depth_checkpoint_tag(checkpoint, t_env, 4)))
    return reused


def remaining_depth_jobs(t_env, recorded_rows, checkpoint=None):
    recorded = list(recorded_rows) + _reuse_final_as_depth(recorded_rows, t_env, checkpoint)
    jobs = []
    for depth in DEPTH_SWEEP_DEPTHS:
        jobs.extend(remaining_jobs(depth_jobs(t_env, checkpoint, depth), recorded))
    return jobs


def depth_eval_finished(output, method, run, seed, t_env, checkpoint=None):
    from open_score.utils.logging import read_records, unique_episodes
    existing = unique_episodes(read_records(output, "episodes", run=run, method=method, seed=seed))
    return not remaining_depth_jobs(t_env, existing, checkpoint)


def infer_best_t_env(output, method, run, seed, fallback=None, rows=None):
    from open_score.utils.logging import read_records
    source = rows if rows is not None else read_records(output, "episodes", run=run, method=method, seed=seed)
    found = [parse_checkpoint_t_env(row.get("checkpoint")) for row in source]
    found = [value for value in found if value is not None]
    if found:
        return max(found)
    return None if fallback is None else int(fallback)


def compute_nds(rho, rho_random, rho_rule):
    if any(value is None for value in (rho, rho_random, rho_rule)):
        return None
    denominator = float(rho_random) - float(rho_rule)
    if denominator == 0:
        return None
    return (float(rho_random) - float(rho)) / denominator


def evaluate_checkpoint(method, checkpoint, *, output=None, run=None, seed=None,
                        device="cpu", on_progress=None, stop_requested=None):
    """Evaluate a retained checkpoint through the same rule-policy episode path."""
    import torch
    from open_score.algos import load_policy
    from open_score.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records, unique_episodes
    from .anchors import BLUE_STRATEGY
    from .report import refresh_report

    run = FORMAL_RUN if run is None else run
    output = Path(DEFAULT_OUTPUT if output is None else output)
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actual_seed = int(saved["config"]["seed"])
    if seed is not None and int(seed) != actual_seed:
        raise ValueError("Evaluation seed must identify the checkpoint training seed")
    t_env = int(saved["progress"]["t_env"])
    from open_score.utils.logging import read_latest
    progress_row = read_latest(output, "progress", run=run).get((method, run, actual_seed), {})
    allowed = official_eval_path(checkpoint.parent, progress_row)
    if run == FORMAL_RUN and (allowed is None or checkpoint.resolve() != Path(allowed).resolve()):
        print(f"[{method}/{actual_seed}] refuse unofficial checkpoint {checkpoint.name} (want {OFFICIAL_CHECKPOINT}.pt)", flush=True)
        return dict(status="blocked", completed=0, total=final_episodes(method),
                    reason="not official last-iterate weights")
    if not _formal_eval_allowed(checkpoint.parent, progress_row, t_env, run):
        print(f"[{method}/{actual_seed}] refuse mid-training final_eval (file t_env={t_env})", flush=True)
        return dict(status="blocked", completed=0, total=final_episodes(method),
                    reason="training not finished")
    checkpoint_label = checkpoint_tag(OFFICIAL_CHECKPOINT if run == FORMAL_RUN else checkpoint.stem, t_env)
    del saved
    policy = load_policy(method, checkpoint)
    if hasattr(policy, "set_device"):
        policy.set_device(device)
    elif device != "cpu":
        raise ValueError("This policy adapter supports CPU evaluation only")
    policy_name = f"crossscale_evaluation_{next(_POLICY_IDS)}"
    register_end_to_end_policy("red", policy_name, f"{method} retained checkpoint", lambda _: policy)
    logger = ExperimentLogger(output, method, actual_seed, run)
    existing = unique_episodes(read_records(output, "episodes", run=run, method=method, seed=actual_seed))
    jobs = remaining_jobs(final_jobs(t_env, OFFICIAL_CHECKPOINT, method=method), existing)
    total = final_episodes(method)
    completed = total - len(jobs)
    started, last_report, initial = time.monotonic(), time.monotonic(), completed
    try:
        for job in jobs:
            if stop_requested is not None and stop_requested():
                logger.progress(phase="final_eval", status="stopped", completed=completed, total=total,
                                t_env=t_env, checkpoint=checkpoint_label)
                return dict(status="stopped", completed=completed, total=total)
            red, blue, targets = config_key(job["config"])
            result = run_episode(targets=targets, red=red, blue=blue, seed=job["episode_seed"],
                                 red_strategy={"architecture": "end_to_end", "policy": policy_name},
                                 blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                 task_mode="damage", spatial_dim=2, target_initialization="random",
                                 diagnostics=True, record_events=True,
                                 retain_trajectory=job["retain_trajectory"])
            summary = dict(result["episode_summary"])
            summary.update(job)
            if hasattr(policy, "episode_q_statistics"):
                summary.update(policy.episode_q_statistics())
            if job["retain_trajectory"]:
                if "trajectory" not in result:
                    raise RuntimeError("Selected final-evaluation episode has no retained trajectory")
                logger.trajectories([dict(phase="final_eval", config=job["config"], episode_seed=job["episode_seed"],
                                          checkpoint=checkpoint_label, trajectory=result["trajectory"])])
            logger.episodes([summary])
            completed += 1
            elapsed = time.monotonic() - started
            progress = dict(phase="final_eval", status="running", completed=completed, total=total,
                            eval_completed=completed, eval_total=total, t_env=t_env,
                            budget_steps=t_env, checkpoint=checkpoint_label,
                            remaining_seconds=(total-completed)*elapsed/max(1, completed-initial))
            logger.progress(**progress)
            if on_progress is not None:
                try:
                    on_progress(dict(progress))
                except Exception as error:
                    print(f"[{method}] final_eval progress callback: {type(error).__name__}: {error}", flush=True)
            if completed % 25 == 0:
                print(f"[{method}/{actual_seed}] final_eval {completed}/{total}", flush=True)
            if time.monotonic() - last_report >= 300:
                _safe_refresh(output, run)
                last_report = time.monotonic()
        logger.progress(phase="final_eval", status="complete", completed=completed, total=total,
                        t_env=t_env, checkpoint=checkpoint_label)
    except BaseException as error:
        logger.progress(phase="final_eval", status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        completed=completed, total=total, t_env=t_env, error=str(error), checkpoint=checkpoint_label)
        raise
    finally:
        _safe_refresh(output, run)
    return dict(status="complete", completed=completed, total=total)


def _main1013_rng_state(cuda=False):
    import random
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if cuda else None)


def _main1013_restore_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def _main1013_clone(value):
    """Copy only mutable controller/capture state, never model parameters."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _main1013_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_main1013_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_main1013_clone(item) for item in value)
    return value


def _main1013_append_diagnostic(output, row):
    """Recover an uncommitted partial tail, then append a durable gzip member."""
    import gzip
    import json
    import os
    import zlib
    from open_score.eval.experiment import atomic_json
    from open_score.utils.logging import _encode, _locked_file
    path = Path(output) / "had" / "diagnostics.jsonl.gz"
    offset_path = path.with_name(".diagnostics.offset.json")
    member = gzip.compress((_encode(row) + "\n").encode("utf-8"), compresslevel=3, mtime=0)
    with _locked_file(path, create=True) as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        offset = (int(json.loads(offset_path.read_text(encoding="utf-8"))["complete_bytes"])
                  if offset_path.exists() else 0)
        if not 0 <= offset <= size:
            raise RuntimeError("Diagnostic offset exceeds the existing file; preserving data")
        stream.seek(offset)
        valid_end, started = offset, False
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        pending = b""
        while True:
            chunk = pending or stream.read(65536)
            pending = b""
            if not chunk:
                break
            started = True
            try:
                decoder.decompress(chunk)
            except zlib.error as error:
                raise RuntimeError("Corrupt diagnostic gzip member; preserving data") from error
            if decoder.eof:
                pending = decoder.unused_data
                valid_end = stream.tell() - len(pending)
                decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                started = False
        if started:
            # The CSV episode is committed only after this append returns.
            # Therefore an interrupted final member has no committed result.
            stream.truncate(valid_end)
        stream.seek(0, os.SEEK_END)
        stream.write(member)
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(offset_path, {"complete_bytes": stream.tell()})


class _Main1013EpisodePolicy:
    """Bounded factual capture and same-state probes around the frozen adapter."""
    def __init__(self, base, seed, config, episode_seed, *, capture=False,
                 perturb=False, reset_gru=False):
        self.base, self.seed, self.config = base, seed, config
        self.episode_seed = episode_seed
        self.capture, self.perturb, self.reset_gru = capture, perturb, reset_gru
        self.representations, self.perturbations = [], []

    def reset(self):
        self.representations, self.perturbations = [], []
        self.base.reset()
        agent = self.base.mac.agent
        if hasattr(agent, "disable_capture"):
            agent.disable_capture()
            agent.last_capture.clear()

    def act(self, state, side, action_ids):
        import torch
        agent, mac = self.base.mac.agent, self.base.mac
        if self.reset_gru:
            mac.hidden_states = torch.zeros_like(mac.hidden_states)
        if self.capture:
            # Keep all live observers, but discard the much larger H capture
            # after this actual decision rather than retaining it for a battle.
            agent.enable_capture(max_decisions=1, max_observers=int(self.base.args.n_agents))
        previous = mac.hidden_states.clone() if self.perturb else None
        rng_before = _main1013_rng_state(self.base.device == "cuda") if self.perturb else None
        actions = self.base.act(state, side, action_ids)
        if self.capture:
            for captured in agent.last_capture:
                self.representations.append({
                    "step": int(state.step),
                    **{key: captured[key].tolist() for key in ("observer_ids", "u_t", "h_t", "h_prev")},
                })
            agent.last_capture.clear()
            agent.disable_capture()
        if self.perturb:
            self.perturbations.append(self._probe(state.step, previous, rng_before))
        return actions

    def _probe(self, step, previous, rng_before):
        import numpy as np
        import torch
        mac, agent = self.base.mac, self.base.mac.agent
        branch = agent.global_net
        rng_after = _main1013_rng_state(self.base.device == "cuda")
        fields = (
            (mac, ("hidden_states", "_decision_q", "_decision_hidden", "_episode_depth")),
            (agent, ("_capture_limits", "last_capture", "_capture_real_decision", "_capture_time", "cycle_depth")),
            (branch, ("_capture_read_indices", "captured_reads", "captured_alpha",
                      "captured_read_attn", "last_alpha", "last_self_attn")),
        )
        retained = [(obj, {key: (hasattr(obj, key), _main1013_clone(getattr(obj, key, None)))
                           for key in names}) for obj, names in fields]
        original_q = mac._decision_q.clone()
        original_hidden = mac.hidden_states.clone()
        alive = ~self.base.batch["entity_mask"][:, step, :previous.shape[1]].bool()
        direction_seed = int(np.random.SeedSequence([
            self.seed, *self.config, self.episode_seed, int(step)]).generate_state(1)[0])
        generator = torch.Generator(device="cpu").manual_seed(direction_seed)
        direction = torch.randn(previous.shape[-1], generator=generator, dtype=torch.float32)
        direction = (direction / direction.square().mean().sqrt()).to(previous.device)
        rms = previous.square().mean(dim=-1, keepdim=True).sqrt()
        delta = (0.01 * rms * direction).masked_fill(~alive.unsqueeze(-1), 0)
        try:
            agent.disable_capture()
            mac.hidden_states = previous + delta
            _main1013_restore_rng(rng_before)
            probe_q = mac.forward(self.base.batch, t=int(step), test_mode=True)[0]
            legal = self.base.batch["avail_actions"][:, step].bool()
            factual_actions = original_q.masked_fill(~legal, -torch.inf).argmax(-1)
            probe_actions = probe_q.masked_fill(~legal, -torch.inf).argmax(-1)
            count = int(alive.sum().item())
            changed = int(((factual_actions != probe_actions) & alive).sum().item())
            q_delta = (probe_q - original_q)[alive]
            h_delta = (mac.hidden_states - original_hidden)[alive]
            return dict(step=int(step), direction_seed=direction_seed, relative_rms=0.01,
                        alive=count, changed_actions=changed,
                        action_change_fraction=changed / max(count, 1),
                        perturbation_rms=float(delta[alive].square().mean().sqrt()) if count else 0.0,
                        q_l2_change=float(q_delta.norm(dim=-1).mean()) if count else 0.0,
                        h_l2_change=float(h_delta.norm(dim=-1).mean()) if count else 0.0)
        finally:
            for obj, values in retained:
                for key, (existed, value) in values.items():
                    if existed:
                        setattr(obj, key, value)
                    elif hasattr(obj, key):
                        delattr(obj, key)
            _main1013_restore_rng(rng_after)


def evaluate_main1013_checkpoint(method, checkpoint, *, output, seed, pool, config,
                                 episode_seeds, device="cpu", on_progress=None,
                                 stop_requested=None):
    """Evaluate an assigned pending subset; the parent's EpisodeIndex deduplicates.

    Identity excludes device and scheduling. Only the seven newly trained arms
    are eligible, and diagnostic forwards never advance the factual simulator.
    """
    import os
    import random
    import numpy as np
    import torch
    from open_score.algos import MAIN1013_OVERRIDES, load_policy
    from open_score.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import ExperimentLogger
    from .anchors import BLUE_STRATEGY

    output, checkpoint = Path(output), Path(checkpoint)
    actual_config = config_key(config)
    scenes = tuple(int(value) for value in episode_seeds)
    reset_configs = ((10, 10, 2), (50, 50, 2), (30, 30, 12))
    supp_methods = ("regir_nomem_r0", "regir_nomem_r1", "regir_nomem_norefil")
    expected_profile = "main1013_supp" if method in supp_methods else "main1013"
    if method not in MAIN1013_OVERRIDES or int(seed) not in (0, 1, 2):
        raise ValueError("Only the seven main1013 arms and seeds 0/1/2 are eligible")
    if output.name != expected_profile or not checkpoint.resolve().is_relative_to(output.resolve()):
        raise ValueError("Checkpoint and output must belong to this arm's new campaign")
    if actual_config not in FINAL_CONFIGS or pool not in ("formal", "confirm", "gru_reset"):
        raise ValueError("Unknown main1013 evaluation pool or configuration")
    allowed = (range(9000, 9300) if pool == "formal" else
               range(110000, 110300) if pool == "confirm" else range(9000, 9010))
    if len(set(scenes)) != len(scenes) or any(value not in allowed for value in scenes):
        raise ValueError("Scene seeds must be a unique subset of the frozen pool")
    if pool == "gru_reset" and (method != "regir_nomem_memtoken" or actual_config not in reset_configs):
        raise ValueError("GRU reset is limited to E1 and its three approved configurations")
    if device not in ("cpu", "cuda"):
        raise ValueError("Evaluation device must be cpu or cuda")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES") if device == "cuda" else None
    if device == "cuda" and physical_gpu not in ("0", "1"):
        raise ValueError("CUDA evaluation needs exactly one permitted visible GPU (0 or 1)")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg, state = saved["config"], saved["progress"]
    t_env = int(state["t_env"])
    if (checkpoint.name != "final.pt" or cfg.get("method") != method
            or int(cfg.get("seed", -1)) != int(seed) or cfg.get("profile") != expected_profile
            or cfg.get("env") != "had" or int(cfg.get("t_max", 0)) != 1_000_000
            or t_env < 1_000_000 or state.get("status") != "completed"):
        raise ValueError("Evaluation requires this new arm's completed 1M-step final checkpoint")
    del saved
    phase = {"formal": "final_eval", "confirm": "confirm_eval", "gru_reset": "mechanism_eval"}[pool]
    protocol = "main1013_memory_placement_v1"
    identity = dict(checkpoint_id=f"main1013/{method}/seed_{int(seed)}/final@{t_env}",
                    checkpoint=f"final@{t_env}", source_version=output.name, protocol=protocol,
                    arm=pool, phase=phase, config=config_dict(actual_config),
                    eval_device=device, physical_gpu=physical_gpu, t_env=t_env, eval_point=50)
    (output / "had").mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(output, method, seed, "train", env="had")
    logger.version = output.name
    completed, damage_sum, measured_seconds = 0, 0.0, 0.0
    total = len(scenes)

    def result(status):
        return dict(status=status, completed=completed, total=total, D_sum=damage_sum,
                    mean_D=damage_sum / completed if completed else None,
                    measured_episodes=completed, measured_seconds=measured_seconds)

    if not scenes:
        return result("complete")
    policy = load_policy(method, checkpoint)
    policy.mac.agent.float().eval()
    if policy.mixer is not None:
        policy.mixer.float().eval()
    policy.set_device(device)
    policy.set_eval_depth(int(cfg.get("global_eval_depth", 4)))
    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    token_methods = ("regir_nomem_memtoken", "regir_nomem_memtoken_sg")
    diagnostic_configs = VALIDATION_CONFIGS + ((15, 15, 2), (30, 30, 2), (50, 50, 2), (30, 30, 12))
    proxy = None
    policy_name = f"main1013_evaluation_{next(_POLICY_IDS)}"
    register_end_to_end_policy("red", policy_name, f"{method} frozen final", lambda _seed: proxy)
    try:
        for scene in scenes:
            if stop_requested is not None and stop_requested():
                logger.progress(**identity, **result("stopped"))
                return result("stopped")
            policy_seed = int(np.random.SeedSequence([int(seed), *actual_config, scene]).generate_state(1)[0])
            random.seed(policy_seed)
            np.random.seed(policy_seed)
            torch.manual_seed(policy_seed)
            diagnostic = (pool == "formal" and scene < 9010 and (
                (actual_config in diagnostic_configs and method in (*token_methods, "transfqmix_aligned"))
                or (method == "regir_nomem_memtoken" and actual_config in reset_configs)))
            proxy = _Main1013EpisodePolicy(policy, int(seed), actual_config, scene,
                capture=diagnostic and method in token_methods and actual_config in diagnostic_configs,
                perturb=diagnostic and method == "regir_nomem_memtoken" and actual_config in reset_configs,
                reset_gru=pool == "gru_reset")
            started = time.monotonic()
            with torch.inference_mode():
                episode = run_episode(targets=actual_config[2], red=actual_config[0], blue=actual_config[1],
                    seed=scene, red_strategy={"architecture": "end_to_end", "policy": policy_name},
                    blue_strategy=BLUE_STRATEGY, max_steps=100, record=False, task_mode="damage",
                    spatial_dim=2, target_initialization="random", diagnostics=True,
                    record_events=True, retain_trajectory=diagnostic)
            wall_seconds = time.monotonic() - started
            summary = dict(episode["episode_summary"])
            summary.update(identity, episode_seed=scene, policy_seed=policy_seed,
                           episode_wall_seconds=wall_seconds)
            summary.update(policy.episode_q_statistics())
            if diagnostic:
                _main1013_append_diagnostic(output, dict(
                    **identity, env="had", method=method, seed=int(seed), episode_seed=scene,
                    policy_seed=policy_seed, D=float(summary["D"]),
                    representation_space="fused_temporal_input", representations=proxy.representations,
                    perturbations=proxy.perturbations,
                    damage_curve=[dict(step=int(frame["step"]), step_damage=float(frame["step_damage"]))
                                  for frame in episode["trajectory"]],
                    reference_status={"SCOPE": "reference_missing", "Memory query": "reference_missing"}))
            # Diagnostic data reaches disk first: a committed factual row can
            # never claim its bounded capture exists when it was not written.
            logger.episodes([summary])
            completed += 1
            damage_sum += float(summary["D"])
            measured_seconds += wall_seconds
            progress = dict(**result("complete" if completed == total else "running"),
                            episode_seed=scene, D=float(summary["D"]), episode_wall_seconds=wall_seconds)
            if on_progress is not None:
                on_progress(progress)
            proxy.base = None
            proxy.representations.clear()
            proxy.perturbations.clear()
        logger.progress(**identity, **result("complete"))
        return result("complete")
    except BaseException as error:
        logger.progress(**identity, **result("failed"), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if proxy is not None:
            proxy.base = None
            proxy.representations.clear()
            proxy.perturbations.clear()
        agent = policy.mac.agent
        if hasattr(agent, "disable_capture"):
            agent.disable_capture()
            agent.last_capture.clear()
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32


def evaluate_depth_sweep(method, checkpoint, *, output=None, run=None, seed=None,
                         device="cpu", on_progress=None, stop_requested=None):
    """1:1 equal-scale sweep over cycle depths R=1..6 for a retained checkpoint."""
    import torch
    from open_score.algos import load_policy
    from open_score.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records, unique_episodes
    from .anchors import BLUE_STRATEGY
    from .report import refresh_report, refresh_report_async

    if method not in CYCLE_SERIES_METHODS:
        raise ValueError(f"{method} has no cycle depth to sweep")
    run = FORMAL_RUN if run is None else run
    output = Path(DEFAULT_OUTPUT if output is None else output)
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actual_seed = int(saved["config"]["seed"])
    if seed is not None and int(seed) != actual_seed:
        raise ValueError("Evaluation seed must identify the checkpoint training seed")
    t_env = int(saved["progress"]["t_env"])
    from open_score.utils.logging import read_latest
    progress_row = read_latest(output, "progress", run=run).get((method, run, actual_seed), {})
    allowed = official_eval_path(checkpoint.parent, progress_row)
    if run == FORMAL_RUN and (allowed is None or checkpoint.resolve() != Path(allowed).resolve()):
        print(f"[{method}/{actual_seed}] refuse unofficial depth checkpoint {checkpoint.name}", flush=True)
        return dict(status="blocked", completed=0, total=depth_eval_total(),
                    reason="not official last-iterate weights")
    if not _formal_eval_allowed(checkpoint.parent, progress_row, t_env, run):
        print(f"[{method}/{actual_seed}] refuse mid-training depth_eval (file t_env={t_env})", flush=True)
        return dict(status="blocked", completed=0, total=depth_eval_total(),
                    reason="training not finished")
    del saved
    policy = load_policy(method, checkpoint)
    if hasattr(policy, "set_device"):
        policy.set_device(device)
    elif device != "cpu":
        raise ValueError("This policy adapter supports CPU evaluation only")
    policy_name = f"crossscale_depth_{next(_POLICY_IDS)}"
    register_end_to_end_policy("red", policy_name, f"{method} cycle-depth sweep", lambda _: policy)
    logger = ExperimentLogger(output, method, actual_seed, run)
    existing = unique_episodes(read_records(output, "episodes", run=run, method=method, seed=actual_seed))
    jobs = remaining_depth_jobs(t_env, existing, OFFICIAL_CHECKPOINT)
    total = depth_eval_total()
    completed = total - len(jobs)
    started, last_report, initial = time.monotonic(), time.monotonic(), completed
    last_depth = None
    print(f"[{method}/{actual_seed}] depth_eval {completed}/{total}", flush=True)

    def report(status, depth=None, extra=None, persist=True):
        progress = dict(phase="depth_eval", status=status, completed=completed, total=total,
                        eval_completed=completed, eval_total=total, t_env=t_env, budget_steps=t_env,
                        checkpoint=depth_checkpoint_tag(OFFICIAL_CHECKPOINT, t_env, depth or last_depth or 4),
                        cycle_depth=None if depth is None and last_depth is None else int(depth or last_depth))
        if extra:
            progress.update(extra)
        if persist:
            logger.progress(**progress)
            if on_progress is not None:
                try:
                    on_progress(dict(progress))
                except Exception as error:
                    print(f"[{method}] depth_eval progress callback: {type(error).__name__}: {error}", flush=True)
        return progress

    try:
        if not jobs:
            report("complete", DEPTH_SWEEP_DEPTHS[-1])
            return dict(status="complete", completed=completed, total=total)
        for job in jobs:
            if stop_requested is not None and stop_requested():
                report("stopped", job.get("cycle_depth"))
                return dict(status="stopped", completed=completed, total=total)
            depth = int(job["cycle_depth"])
            if depth != last_depth:
                if last_depth is not None:
                    refresh_report_async(output, run=run)
                    last_report = time.monotonic()
                if hasattr(policy, "set_eval_depth"):
                    policy.set_eval_depth(depth)
                last_depth = depth
                print(f"[{method}/{actual_seed}] depth_eval R{depth} {completed}/{total}", flush=True)
            red, blue, targets = config_key(job["config"])
            result = run_episode(targets=targets, red=red, blue=blue, seed=job["episode_seed"],
                                 red_strategy={"architecture": "end_to_end", "policy": policy_name},
                                 blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                 task_mode="damage", spatial_dim=2, target_initialization="random",
                                 diagnostics=True, record_events=True, retain_trajectory=False)
            summary = dict(result["episode_summary"])
            summary.update(job)
            summary.pop("cycle_depth", None)
            summary.pop("retain_trajectory", None)
            if hasattr(policy, "episode_q_statistics"):
                summary.update(policy.episode_q_statistics())
            logger.episodes([summary])
            completed += 1
            elapsed = time.monotonic() - started
            remaining = (total - completed) * elapsed / max(1, completed - initial)
            persist = completed % 10 == 0 or completed == total
            report("running", depth, extra=dict(remaining_seconds=remaining), persist=persist)
            if persist:
                label = config_label(job["config"])
                print(f"[{method}/{actual_seed}] depth_eval R{depth} {completed}/{total} {label}", flush=True)
            if time.monotonic() - last_report >= 60:
                refresh_report_async(output, run=run)
                last_report = time.monotonic()
        report("complete", last_depth)
    except BaseException as error:
        logger.progress(phase="depth_eval", status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        completed=completed, total=total, t_env=t_env, error=str(error),
                        eval_completed=completed, eval_total=total,
                        checkpoint=depth_checkpoint_tag(OFFICIAL_CHECKPOINT, t_env, last_depth or 4),
                        cycle_depth=last_depth)
        raise
    finally:
        _safe_refresh(output, run)
    return dict(status="complete", completed=completed, total=total)


def _smac_eval_runner(saved, config, device="cpu"):
    """Rebuild capacity, retaining all learned parameter shapes and native inputs."""
    import copy
    from types import SimpleNamespace
    from open_score.algos import setup_runtime, make_scheme, LearnerLogger
    setup_runtime()
    from runners.parallel_runner import ParallelRunner
    from open_score.models import build_mac
    cfg = copy.deepcopy(saved["config"])
    cfg.update(device=device, use_cuda=device == "cuda", batch_size_run=1)
    cfg["env_args"].update(pad=(config["N_R"], config["N_B"]), config=config, max_retries=1)
    args = SimpleNamespace(**cfg)
    runner = ParallelRunner(args, LearnerLogger())
    try:
        for key, value in runner.get_env_info().items():
            setattr(args, key, value)
        scheme, groups, preprocess = make_scheme(runner.get_env_info(), multi_task=False)
        model_scheme = copy.deepcopy(scheme)
        model_scheme["entities"]["vshape"] = args.entity_shape
        model_scheme["actions_onehot"] = {"vshape": (args.n_actions,), "group": "agents"}
        mac = build_mac(model_scheme, groups, args)
        mac.agent.load_state_dict(saved["networks"]["agent"])
        if "mac" in saved["networks"]:
            mac.load_state_dict(saved["networks"]["mac"])
        if args.use_cuda:
            mac.cuda()
        mac.eval()
        runner.setup(scheme, groups, preprocess, mac)
        return runner
    except BaseException:
        runner.close_env()
        raise


def evaluate_profile_checkpoint(method, checkpoint, *, output, kind="final", env="had",
                                seed=None, device="cpu", on_progress=None, stop_requested=None,
                                assigned_jobs=None, shard_id=None, resume_shard=False):
    """One fixed checkpoint/arm queue, resumed by immutable result identity."""
    import json
    import os
    import re
    import numpy as np
    import torch
    from .experiment import (checkpoint_info, initialize, evaluation_jobs,
                             remaining_evaluations, result_identity, atomic_json)
    from open_score.utils.logging import ExperimentLogger, read_records
    from open_score.utils.resources import cpu_threads
    cpu_threads()
    torch.set_num_threads(1)
    if device not in ("cpu", "cuda"):
        raise ValueError("Evaluation device must be cpu or cuda with one visible physical GPU")
    if device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise ValueError("CUDA evaluation workers require one permitted physical GPU (0 or 1)")
    if shard_id is not None:
        shard_id = str(shard_id)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", shard_id):
            raise ValueError("Evaluation shard identity must be a safe filename component")
    info = checkpoint_info(checkpoint, method=method, seed=seed, env=env)
    from .experiment import test_depth, EVAL_PHASES, PROFILE
    if kind not in EVAL_PHASES:
        raise ValueError(f"Unknown evaluation kind {kind}")
    if kind in ("depth", "readout", "r1deploy", "gate_depth") and test_depth(method) is None:
        raise ValueError("Depth and readout interventions require a relation-cycle final")
    manifest = initialize(output)
    frozen_jobs = evaluation_jobs(info, kind)
    if assigned_jobs is None:
        existing = read_records(output, "episodes", run="train", env=env, method=method, seed=info["seed"])
        jobs = remaining_evaluations(info, kind, existing,
            read1_equivalent=manifest["mechanisms"]["read1_equivalence_verified"])
        total = len(frozen_jobs)
    else:
        # The sole parent has already removed completed episodes and assigned
        # disjoint cells. Fresh workers do not each rescan the shared history.
        canonical = {result_identity(job): job for job in frozen_jobs}
        identities = [result_identity(job) for job in assigned_jobs]
        if len(set(identities)) != len(identities) or any(key not in canonical for key in identities):
            raise ValueError("Assigned evaluation jobs must be a unique subset of this frozen checkpoint/arm")
        jobs = [dict(canonical[key]) for key in identities]
        total = len(jobs)
        if resume_shard:
            existing = read_records(output, "episodes", run="train", env=env, method=method, seed=info["seed"])
            pending = {result_identity(job) for job in remaining_evaluations(info, kind, existing,
                read1_equivalent=manifest["mechanisms"]["read1_equivalence_verified"])}
            jobs = [job for job in jobs if result_identity(job) in pending]
    completed = initial = total - len(jobs)
    logger = ExperimentLogger(output, method, info["seed"], "train", env=env)
    start = time.monotonic()
    policy = runner = None
    runner_config = None
    failed = {}
    legacy_failure_path = Path(checkpoint).parent / "evaluation_failures.json"
    failure_path = (legacy_failure_path if shard_id is None else
                    legacy_failure_path.with_name(f"evaluation_failures.{shard_id}.json"))
    if env == "had":
        from open_score.algos import load_policy
        from open_score.rules import register_end_to_end_policy, run_episode
        from .anchors import BLUE_STRATEGY
        policy = load_policy(method, checkpoint)
        policy.set_device(device)
        policy_name = f"{PROFILE}_{next(_POLICY_IDS)}"
        register_end_to_end_policy("red", policy_name, f"{PROFILE} frozen final", lambda _: policy)
        branch = getattr(policy.mac.agent, "global_net", None)
        default_depth = test_depth(method)
    else:
        from open_score.envs.smacv2_env import generate_registered_scenes
        scenes = generate_registered_scenes(output)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        scene_ids = {f"{job['config']['N_R']}v{job['config']['N_B']}.s{job['episode_seed']}" for job in jobs}
        # Resuming with a different shard count must not reset a scene's
        # registered retry limit. Each current worker still writes its own file.
        for path in (legacy_failure_path, *legacy_failure_path.parent.glob("evaluation_failures.part*.json")):
            if path.exists():
                old = json.loads(path.read_text())
                if old.get("checkpoint_id") == info["checkpoint_id"]:
                    for scene_id, failure in old.get("scenes", {}).items():
                        if (scene_id in scene_ids and
                                failure.get("attempts", 0) >= failed.get(scene_id, {}).get("attempts", 0)):
                            failed[scene_id] = failure

    def policy_seed(job):
        # Keep policy draws independent of worker order, retries and shard
        # boundaries. CUDA and CPU still use their own RNG implementations.
        return int(np.random.SeedSequence([info["seed"], job["config"]["N_R"],
            job["config"]["N_B"], job["config"]["K"], job["episode_seed"]]).generate_state(1)[0])

    def progress(status, job=None):
        row = dict(phase=f"{kind}_eval", status=status, completed=completed, total=total,
                   t_env=info["t_env"], checkpoint=info["checkpoint"], checkpoint_id=info["checkpoint_id"],
                   eval_device=device, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES") if device == "cuda" else None,
                   shard=shard_id,
                   remaining_seconds=(total-completed)*(time.monotonic()-start)/max(1, completed-initial))
        if job:
            row.update(arm=job["arm"], cycle_depth=job.get("cycle_depth"), readout=job.get("readout"),
                       config=job["config"], episode_seed=job["episode_seed"], policy_seed=policy_seed(job))
        if status != "running" or (completed - initial) % 25 == 0:
            logger.progress(**row)
        if on_progress:
            on_progress(row)
        return dict(status=status, completed=completed, total=total, failed_scenes=len(failed))

    last_job = None
    try:
        for job in jobs:
            last_job = job
            if stop_requested and stop_requested():
                return progress("stopped", job)
            if env == "had":
                if job.get("cycle_depth") is not None:
                    policy.set_eval_depth(job["cycle_depth"])
                elif default_depth is not None:
                    policy.set_eval_depth(default_depth)
                if kind == "readout":
                    policy.mac.agent.global_net.readout_override = job["readout"]
                if branch is not None and getattr(branch, "intent", False):
                    branch.intent_override = job.get("intent_override")
                torch.manual_seed(policy_seed(job))
                with torch.inference_mode():
                    result = run_episode(targets=job["config"]["K"], red=job["config"]["N_R"],
                        blue=job["config"]["N_B"], seed=job["episode_seed"],
                        red_strategy={"architecture": "end_to_end", "policy": policy_name},
                        blue_strategy=BLUE_STRATEGY, max_steps=100, record=False, task_mode="damage",
                        spatial_dim=2, target_initialization="random", diagnostics=True, record_events=True,
                        retain_trajectory=job["retain_trajectory"])
                summary, trajectory = dict(result["episode_summary"]), result.get("trajectory")
                if hasattr(policy, "episode_q_statistics"):
                    summary.update(policy.episode_q_statistics())
            else:
                scene_id = f"{job['config']['N_R']}v{job['config']['N_B']}.s{job['episode_seed']}"
                scene = scenes["scenes"][scene_id]
                job["reset_config"] = scene["reset_config"]
                job["engine_seed"] = scene["engine_seed"]
                if failed.get(scene_id, {}).get("attempts", 0) >= 3:
                    continue
                summary = trajectory = None
                for attempt in range(int(failed.get(scene_id, {}).get("attempts", 0)), 3):
                    try:
                        if runner is None or runner_config != job["config"]:
                            if runner is not None:
                                runner.close_env()
                            runner = _smac_eval_runner(saved, job["config"], device=device)
                            runner_config = dict(job["config"])
                        if job.get("cycle_depth") is not None:
                            runner.mac.set_global_depth(int(job["cycle_depth"]))
                            runner.mac.args.global_eval_depth = int(job["cycle_depth"])
                        torch.manual_seed(policy_seed(job))
                        with torch.inference_mode():
                            _, summaries = runner.run(test_mode=True, jobs=[job])
                        summary = dict(summaries[0])
                        trajectory = runner.last_trajectories[0]
                        failed.pop(scene_id, None)
                        if failure_path.exists():
                            atomic_json(failure_path, dict(checkpoint_id=info["checkpoint_id"], scenes=failed))
                        break
                    except Exception as error:
                        if runner is not None:
                            runner.close_env()
                            runner = None
                        if not (isinstance(error, (EOFError, BrokenPipeError, ConnectionError))
                                or "SMACEnvironmentError" in str(error)
                                or "Environment worker" in str(error)):
                            raise
                        # A protocol/connection failure is not a battle loss.
                        failed[scene_id] = dict(attempts=attempt+1, error=f"{type(error).__name__}: {error}")
                        atomic_json(failure_path, dict(checkpoint_id=info["checkpoint_id"], scenes=failed))
                        if stop_requested and stop_requested():
                            return progress("stopped", job)
                if summary is None:
                    continue
            summary.update({k: v for k, v in job.items() if k != "reset_config"})
            logger.episodes([summary])
            if job["retain_trajectory"] and trajectory is not None:
                logger.trajectories([{**summary, "trajectory": trajectory}])
            completed += 1
            progress("running", job)
            if completed % 25 == 0:
                print(f"[{env}/{method}/{info['seed']}] {kind}_eval {completed}/{total}", flush=True)
        return progress("complete" if completed == total else "incomplete", last_job)
    finally:
        if runner is not None:
            runner.close_env()
