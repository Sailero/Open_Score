"""Decision-time ms/step on the current entity interface, excluding env.step."""
from __future__ import annotations

from pathlib import Path
import time

from open_score.eval.protocol import (DEPTH_SWEEP_DEPTHS, OFFICIAL_CHECKPOINT, TIMING_METHODS,
                                      TIMING_SCALES, config_dict, config_label)
from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, ExperimentLogger, read_records


class _TimedPolicy:
    def __init__(self, inner):
        self.inner = inner
        self.elapsed = 0.0
        self.steps = 0

    def reset(self):
        self.inner.reset()

    def act(self, *args, **kwargs):
        started = time.perf_counter()
        result = self.inner.act(*args, **kwargs)
        self.elapsed += time.perf_counter() - started
        self.steps += 1
        return result

    def episode_q_statistics(self):
        if hasattr(self.inner, "episode_q_statistics"):
            return self.inner.episode_q_statistics()
        return {}


def _param_count(policy):
    total = 0
    for module in (getattr(policy, "mac", None), getattr(policy, "mixer", None)):
        if module is None:
            continue
        parameters = getattr(module, "parameters", None)
        if callable(parameters):
            total += sum(item.numel() for item in parameters())
        elif hasattr(module, "agent"):
            total += sum(item.numel() for item in module.agent.parameters())
    return int(total)


def evaluate_timing(*, output=None, run=None, device="cuda", stop_requested=None,
                    warmup=1, repeats=5):
    from open_score.eval.experiment import is_profile
    if is_profile(DEFAULT_OUTPUT if output is None else output):
        return evaluate_profile_timing(output=DEFAULT_OUTPUT if output is None else output,
                                       stop_requested=stop_requested)
    from open_score.algos import load_policy
    from open_score.eval.anchors import BLUE_STRATEGY
    from open_score.eval.inventory import _gpu_training_busy, run_dir
    from open_score.eval.report import refresh_report
    from open_score.rules import register_end_to_end_policy, run_episode

    if device == "cuda" and _gpu_training_busy():
        return dict(status="blocked", reason="gpu training busy")
    run = FORMAL_RUN if run is None else run
    output = Path(DEFAULT_OUTPUT if output is None else output)
    existing = read_records(output, "timing", run=run) if (output / "timing.csv").exists() else []
    done = {(row.get("method"),
             tuple(int(row["config"][key]) for key in ("N_R", "N_B", "K")) if isinstance(row.get("config"), dict) else None,
             row.get("cycle_depth"))
            for row in existing}
    logger = ExperimentLogger(output, "timing", 0, run)
    written = 0
    try:
        for method in TIMING_METHODS:
            checkpoint = run_dir(output, method, 0) / f"{OFFICIAL_CHECKPOINT}.pt"
            if not checkpoint.exists():
                continue
            policy = load_policy(method, checkpoint)
            if hasattr(policy, "set_device"):
                try:
                    policy.set_device(device)
                except Exception as error:
                    print(f"[timing] {method} set_device({device}) failed: {error}; use cpu", flush=True)
                    device = "cpu"
            params = _param_count(policy)
            depths = DEPTH_SWEEP_DEPTHS if method == "regir" else (4,)
            for scale in TIMING_SCALES:
                for depth in depths:
                    if stop_requested is not None and stop_requested():
                        return dict(status="stopped", written=written)
                    key = (method, scale, depth if method == "regir" else None)
                    if key in done or (method, scale, depth) in done:
                        continue
                    if hasattr(policy, "set_eval_depth"):
                        policy.set_eval_depth(int(depth))
                    timed = _TimedPolicy(policy)
                    name = f"timing_{method}_{scale[0]}_{depth}"
                    register_end_to_end_policy("red", name, "timed policy", lambda inner=timed: inner)
                    red, blue, targets = scale
                    # Warmup is not recorded.
                    for seed in range(warmup):
                        timed.elapsed = timed.steps = 0
                        run_episode(targets=targets, red=red, blue=blue, seed=9200 + seed,
                                    red_strategy={"architecture": "end_to_end", "policy": name},
                                    blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                    task_mode="damage", spatial_dim=2, target_initialization="random",
                                    diagnostics=False, record_events=False, retain_trajectory=False)
                    elapsed = steps = 0
                    for seed in range(repeats):
                        timed.elapsed = timed.steps = 0
                        run_episode(targets=targets, red=red, blue=blue, seed=9210 + seed,
                                    red_strategy={"architecture": "end_to_end", "policy": name},
                                    blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                                    task_mode="damage", spatial_dim=2, target_initialization="random",
                                    diagnostics=False, record_events=False, retain_trajectory=False)
                        elapsed += timed.elapsed
                        steps += timed.steps
                    ms = 1000.0 * elapsed / max(steps, 1)
                    row = dict(method=method, config=config_dict(scale), device=device,
                               cycle_depth=int(depth) if method == "regir" else None,
                               repeat_id=0, ms_per_step=ms, n_steps=steps, params=params)
                    logger.timing([row])
                    written += 1
                    print(f"[timing] {method} {config_label(scale)} R={depth} {ms:.2f} ms/step params={params}",
                          flush=True)
        logger.progress(phase="timing", status="complete", completed=written)
    finally:
        refresh_report(output, run=run)
    return dict(status="complete", written=written)


class _CostStopped(Exception):
    pass


def _cost_stop(stop_requested):
    if stop_requested is not None and stop_requested():
        raise _CostStopped()


def _cost_parameters(policy):
    """Online trainable parameters only; target replicas are not new capacity."""
    actor = {id(p): p for p in policy.mac.parameters() if p.requires_grad}
    online = dict(actor)
    if policy.mixer is not None:
        online.update({id(p): p for p in policy.mixer.parameters() if p.requires_grad})
    return sum(p.numel() for p in actor.values()), sum(p.numel() for p in online.values())


def _cost_state_bank(output):
    """Select two predetermined shared scenes; never collect an environment episode."""
    import json
    from open_score.eval.experiment import checkpoint_info, run_directory
    from open_score.eval.relationship_probe import scene_jobs, selected_times, _read_trajectory
    root = Path(output) / "probe"
    manifest_path, complete_path = root / "manifest.json", root / "collection_complete.json"
    if not manifest_path.exists() or not complete_path.exists():
        raise FileNotFoundError("Complete shared probe state bank is required before M4 timing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("trajectories") != 500 or any(complete.get(k) != v for k, v in manifest.items()):
        raise ValueError("Probe completion marker does not match the frozen public state bank")
    for method in ("regir", "refil"):
        path = run_directory(output, method, 0) / "final.pt"
        if not path.is_file():
            # Imported no-weight rows still pin identity in the frozen probe manifest.
            continue
        info = checkpoint_info(path, method=method, seed=0, env="had")
        if manifest.get("behavior_checkpoint_ids", {}).get(method) != info["checkpoint_id"]:
            raise ValueError("M4 public trajectory behavior checkpoint identity is stale")
    selected = []
    for n in (10, 50):
        job = next(j for j in scene_jobs() if tuple(j["config"]) == (n, n, 2) and j["episode_index"] == 0)
        path = root / "trajectories" / f"scene_{job['episode_seed']}.json.gz"
        trajectory = _read_trajectory(path)
        if (trajectory.get("job") != job or trajectory.get("checkpoint_id") != manifest["behavior_checkpoint_ids"][job["behavior"]]
                or len(trajectory.get("frames", [])) != trajectory.get("steps")):
            raise ValueError(f"Incomplete/mismatched public trajectory: {path}")
        times = selected_times(trajectory["frames"])
        if not times:
            raise ValueError(f"No valid public decision state in {path}")
        index = int(times[len(times) // 2])
        state = trajectory["frames"][index]["state"]
        selected.append(dict(config=(n, n, 2), trajectory=trajectory, index=index,
                             scene_id=job["episode_seed"], physical_step=int(state["step"]),
                             live_red=sum(e["health"] > 0 for e in state["red"]),
                             live_blue=sum(e["health"] > 0 for e in state["blue"]), source=str(path)))
    return selected


def _cost_copy(value):
    """Clone input history before synchronization, outside the timed interval."""
    import torch
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _cost_copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cost_copy(v) for v in value)
    if isinstance(value, list):
        return [_cost_copy(v) for v in value]
    return value


def _cost_prepare(policy, selection, stop_requested=None):
    """Replay recorded actions once and freeze the input plus pre-decision memory.

    Controller tensor assembly/relative encoding remains in the timed actor
    path for every architecture. DecisionState conversion, prefix replay,
    EpisodeBatch updates, action selection, mixer and decoding are excluded.
    """
    from open_score.envs import DecisionState
    from open_score.eval.relationship_probe import teacher_forced_act
    import torch
    policy.reset()
    if getattr(policy.args, "global_branch", None) in ("cycle", "slot", "feedback"):
        depth = selection.get("depth")
        if depth is None:
            name = str(getattr(policy.args, "method", ""))
            depth = 1 if name.startswith("regir_r1") or name.startswith("regir_r0") else 4
        policy.set_eval_depth(int(depth))
    agent = policy.mac.agent
    if hasattr(agent, "disable_capture"):
        agent.disable_capture()
    agent.capture_attention = False
    if hasattr(agent, "global_net"):
        agent.global_net.capture_attention = False
        if hasattr(agent.global_net, "self_attn"):
            agent.global_net.self_attn.capture_attention = False
    # All cost methods are non-hierarchical; explicit checks prevent silently
    # omitting a future method's deployment-time allocation/coach network.
    if getattr(policy.mac, "use_alloc", False) or getattr(policy.mac, "use_copa", False):
        raise ValueError("M4 actor timing has no contract for hierarchical allocation/coach state")
    capture = {}
    original_forward = policy.mac.forward
    def capture_forward(batch, t, *args, **kwargs):
        if int(t) == selection["physical_step"]:
            capture.update(batch=batch, t=int(t), hidden=_cost_copy(policy.mac.hidden_states),
                           depth=_cost_copy(getattr(policy.mac, "_episode_depth", None)))
        return original_forward(batch, t, *args, **kwargs)
    policy.mac.forward = capture_forward
    # Prefix reconstruction does not need diagnostic joint Q or a mixer.
    evaluation_values = getattr(policy.mac, "evaluation_values", None)
    if evaluation_values is not None:
        policy.mac.evaluation_values = lambda batch, t, actions, active, mixer: {
            "q_tot": [None] * len(active), "q_i": [None] * len(active)}
    try:
        with torch.no_grad():
            for frame in selection["trajectory"]["frames"][:selection["index"] + 1]:
                _cost_stop(stop_requested)
                state = DecisionState.from_dict(frame["state"])
                teacher_forced_act(policy, state, frame["actions"], frame["action_ids"])
    finally:
        policy.mac.forward = original_forward
        if evaluation_values is not None:
            policy.mac.evaluation_values = evaluation_values
    if "batch" not in capture:
        raise ValueError("Public prefix did not reach the predetermined M4 decision")
    if capture["batch"].batch_size != 1:
        raise ValueError("M4 fixes the episode batch size at one full team")
    def restore():
        policy.mac.hidden_states = _cost_copy(capture["hidden"])
        if hasattr(policy.mac, "_episode_depth"):
            policy.mac._episode_depth = _cost_copy(capture["depth"])
    def forward():
        return policy.mac.forward(capture["batch"], capture["t"], acting=True, test_mode=True)
    return restore, forward


def _cost_hardware(physical_gpu, device):
    """Snapshot competing compute allocations outside the measured interval."""
    import csv
    from datetime import datetime, timezone
    import os
    import subprocess
    import torch
    snapshot = dict(physical_gpu=physical_gpu, logical_device=str(device),
                    hardware_model=torch.cuda.get_device_name(device),
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    other_compute_memory_mib=None,
                    memory_scope="nvidia-smi compute-app allocations excluding this PID; graphics/driver memory not attributed")
    try:
        def query(fields, scope):
            result = subprocess.run(["nvidia-smi", f"--id={physical_gpu}", f"--query-{scope}={fields}",
                                     "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                    text=True, timeout=10)
            return [row for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True) if row]
        gpu = query("uuid,memory.total,memory.used", "gpu")[0]
        snapshot.update(hardware_uuid=gpu[0], total_memory_mib=float(gpu[1]),
                        total_used_memory_mib=float(gpu[2]))
        processes = [dict(pid=int(pid), used_memory_mib=float(used))
                     for pid, used in query("pid,used_gpu_memory", "compute-apps")]
        others = [row for row in processes if row["pid"] != os.getpid()]
        snapshot.update(other_compute_process_count=len(others),
                        other_compute_memory_mib=sum(row["used_memory_mib"] for row in others),
                        own_compute_memory_mib=sum(row["used_memory_mib"] for row in processes if row["pid"] == os.getpid()))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        # Never substitute zero when the driver cannot attribute memory.
        snapshot["memory_query_error"] = str(error)
    return snapshot


def evaluate_profile_timing(output, stop_requested=None, on_progress=None):
    """M4: 8 methods × 3 seeds × 2 scales on one registered physical GPU.

    The caller selects the fixed measurement GPU (0 or 1), owns its lock and excludes concurrent
    training. This function deliberately does not acquire that lock again.
    One final-identity-bound aggregate is appended per method/seed/scale.
    """
    import json
    import math
    import os
    import numpy as np
    import torch
    from open_score.algos import load_policy
    from open_score.eval.experiment import COST_METHODS, SEEDS, checkpoint_info, cost_depths, run_directory
    from open_score.utils.logging import iter_records
    output = Path(output)
    warmup, repeats = 50, 200
    total = 0
    written, completed = 0, 0
    infos = {}
    try:
        _cost_stop(stop_requested)
        # Resolve all required final artifacts before allocating a GPU policy.
        for method in COST_METHODS:
            for seed in SEEDS:
                path = run_directory(output, method, seed) / "final.pt"
                if not path.exists():
                    continue
                infos[method, seed] = checkpoint_info(path, method=method, seed=seed, env="had")
        if not infos:
            raise FileNotFoundError("M4 has no measurable finals yet")
        selections = _cost_state_bank(output)
        resources = json.loads((output / "experiment.json").read_text(encoding="utf-8")).get("resources", {})
        physical_gpu = resources.get("measurement_gpu")
        if type(physical_gpu) is not int or physical_gpu < 0:
            raise ValueError("M4 requires a registered resources.measurement_gpu >= 0")
    except _CostStopped:
        return dict(status="stopped", completed=0, total=total, written=0)
    except (FileNotFoundError, ValueError) as error:
        return dict(status="blocked", completed=0, total=total, written=0, reason=str(error))
    wanted = {("had", info["checkpoint_id"], (n, n, 2), f"cost:R{depth}" if depth else "cost")
              for (method, seed), info in infos.items()
              for n in (10, 50) for depth in cost_depths(method)}
    done = set()
    for row in iter_records(output, "timing", run="train", env="had"):
        cfg = row.get("config")
        if not cfg:
            continue
        key = (row.get("env"), row.get("checkpoint_id"), tuple(int(cfg[k]) for k in ("N_R", "N_B", "K")), row.get("arm"))
        measures = ("ms_per_step", "p25_ms", "p75_ms", "p95_ms", "actor_params", "training_params")
        if (key in wanted and row.get("device") == "cuda:0" and row.get("physical_gpu") == physical_gpu
                and row.get("n_steps") == repeats
                and all(row.get(k) is not None and math.isfinite(float(row[k])) for k in measures)):
            done.add(key)
    total = len(wanted)
    completed = len(done)
    if total and completed == total:
        return dict(status="complete", completed=completed, total=total, written=0, reused=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() != str(physical_gpu):
        return dict(status="blocked", completed=completed, total=total, written=0,
                    reason=f"M4 requires CUDA_VISIBLE_DEVICES={physical_gpu} (registered physical GPU), with the caller holding its queue lock")
    if not torch.cuda.is_available():
        return dict(status="blocked", completed=completed, total=total, written=0,
                    reason=f"M4 registered physical GPU{physical_gpu} is unavailable; no CPU fallback")
    device = torch.device("cuda:0")
    old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for method in COST_METHODS:
            for seed in SEEDS:
                _cost_stop(stop_requested)
                info = infos.get((method, seed))
                if info is None:
                    continue
                pending = []
                for selection in selections:
                    for depth in cost_depths(method):
                        arm = f"cost:R{depth}" if depth else "cost"
                        if ("had", info["checkpoint_id"], selection["config"], arm) not in done:
                            pending.append(dict(selection, depth=depth, arm=arm))
                if not pending:
                    continue
                policy = load_policy(method, info["path"])
                policy.mac.agent.float().eval()
                if policy.mixer is not None:
                    policy.mixer.float().eval()
                actor_params, training_params = _cost_parameters(policy)
                policy.set_device(str(device))
                logger = ExperimentLogger(output, method, seed, "train", env="had")
                restore = forward = None
                try:
                    for selection in pending:
                        _cost_stop(stop_requested)
                        restore, forward = _cost_prepare(policy, selection, stop_requested)
                        contract = dict(scene_id=selection["scene_id"], physical_step=selection["physical_step"],
                                        **_cost_hardware(physical_gpu, device),
                                        source=selection["source"], live_red=selection["live_red"], live_blue=selection["live_blue"],
                                        episode_batch=1, precision="float32", tf32=False, warmup=warmup, repeats=repeats,
                                        padded_agents=int(policy.args.n_agents), padded_entities=int(policy.args.n_entities),
                                        hidden="same model-specific h_prev reconstructed from common real action prefix before every call",
                                        measured="full-team mac.forward including model tensor encoding; excluding environment, state/batch reconstruction, action decoding, mixer",
                                        parameters="trainable online actor + online mixer, unique objects; target replicas excluded",
                                        timing="host perf_counter + synchronize cuda:0 before/after each forward; restore outside timer")
                        elapsed = []
                        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                            for index in range(warmup + repeats):
                                _cost_stop(stop_requested)
                                restore()
                                torch.cuda.synchronize(device)
                                started = time.perf_counter()
                                result = forward()
                                torch.cuda.synchronize(device)
                                milliseconds = (time.perf_counter() - started) * 1000.0
                                del result
                                if index >= warmup:
                                    elapsed.append(milliseconds)
                        median, p25, p75, p95 = map(float, np.percentile(elapsed, [50, 25, 75, 95]))
                        row = dict(env="had", checkpoint_id=info["checkpoint_id"], arm=selection["arm"],
                                   config=config_dict(selection["config"]),
                                   device="cuda:0", physical_gpu=physical_gpu,
                                   cycle_depth=selection.get("depth"),
                                   readout="learned", repeat_id=0, ms_per_step=median, n_steps=repeats,
                                   params=training_params, actor_params=actor_params, training_params=training_params,
                                   p25_ms=p25, p75_ms=p75, p95_ms=p95)
                        logger.progress(phase="timing", arm=selection["arm"], status="measured",
                                        checkpoint_id=info["checkpoint_id"],
                                        config=row["config"], t_env=info["t_env"], measurement_contract=contract)
                        logger.timing([row])
                        written += 1
                        completed += 1
                        if on_progress is not None:
                            on_progress(dict(phase="timing", method=method, seed=seed, completed=completed,
                                             total=total, config=row["config"], median_ms=median, physical_gpu=physical_gpu))
                        print(f"[M4] {method} s{seed} {_config_cost_label(selection['config'])}: "
                              f"{median:.3f} [{p25:.3f}, {p75:.3f}] p95={p95:.3f} ms; actor={actor_params}, online={training_params}", flush=True)
                finally:
                    # The closures retain this model and its prepared batch.
                    # Release them before loading the next checkpoint.
                    restore = forward = None
                    del policy
                    torch.cuda.empty_cache()
        return dict(status="complete", completed=completed, total=total, written=written)
    except _CostStopped:
        return dict(status="stopped", completed=completed, total=total, written=written)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32


def _config_cost_label(config):
    return f"{config[0]}v{config[1]} K{config[2]}"
