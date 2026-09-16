"""Shared ALMA training entry and frozen-policy interface for HAD experiments."""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN, VERSION

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent
UPSTREAM = Path(__file__).resolve().parent / "pymarl"
METHODS = ("b0_qmix", "b2_qmix_atten", "refil", "dcg", "gnn_qmix", "spectra", "alma")
PROBE_METHODS = ("alma_fullobs", "alma_blue", "alma_event", "alma_nomask")
V4_METHODS = ("refil_local_mild", "refil_local_mid", "refil_count")
V5_GLOBAL_METHODS = ("refil_cycle", "refil_card", "refil_feedback", "refil_slot")
V5_METHODS = ("refil_count",) + V5_GLOBAL_METHODS
POLICY_METHODS = METHODS + PROBE_METHODS + tuple(dict.fromkeys((*V4_METHODS, *V5_METHODS)))
V4_OVERRIDES = {
    "refil_local_mild": {
        "imagine_group": "mixed_distance",
        "imagine_group_eta": 0.15,
        "imagine_group_radius": 0.4,
    },
    "refil_local_mid": {
        "imagine_group": "mixed_distance",
        "imagine_group_eta": 0.30,
        "imagine_group_radius": 0.4,
    },
    "refil_count": {
        "imagine_group": "original",
        "count_cond": "phi2",
    },
}
V5_OVERRIDES = {
    "refil_count": {
        "imagine_group": "original",
        "count_cond": "phi2",
        "count_ln": False,
    },
    "refil_cycle": {
        "imagine_group": "original",
        "count_cond": "phi2",
        "global_branch": "cycle",
        "global_slots": 4,
        "global_depths": [1, 2, 3, 4],
        "global_eval_depth": 4,
        "global_embed_dim": 64,
        "global_ffn_mult": 2,
        "global_n_heads": 4,
    },
    "refil_card": {
        "imagine_group": "original",
        "count_cond": "phi2",
        "global_branch": "card",
        "global_slots": 4,
        "global_depths": [1, 2, 3, 4],
        "global_eval_depth": 4,
        "global_embed_dim": 64,
        "global_ffn_mult": 2,
        "global_n_heads": 4,
    },
    "refil_feedback": {
        "imagine_group": "original",
        "count_cond": "phi2",
        "global_branch": "feedback",
        "global_slots": 4,
        "global_depths": [1, 2, 3, 4],
        "global_eval_depth": 4,
        "global_embed_dim": 64,
        "global_ffn_mult": 2,
        "global_n_heads": 4,
    },
    "refil_slot": {
        "imagine_group": "original",
        "count_cond": "phi2",
        "global_branch": "slot",
        "global_slots": 4,
        "global_depths": [1, 2, 3, 4],
        "global_eval_depth": 4,
        "global_embed_dim": 64,
        "global_ffn_mult": 2,
        "global_n_heads": 4,
    },
}
# Everything else defines the experiment and must match on resume, so that a
# single run never mixes old replay data with changed learning conditions.
RESUME_MUTABLE = frozenset(("resume", "output", "device", "use_cuda",
                            "report_interval", "progress_interval",
                            "implementation_revision", "concurrency",
                            "entity_pad"))
# Keys added after some runs started; missing saved values equal these defaults.
RESUME_DEFAULTS = {"reward_mode": "damage", "friendly_penalty": 1.0,
                   "count_ln": True, "global_branch": None, "global_slots": 4,
                   "global_depths": [1, 2, 3, 4], "global_eval_depth": 4,
                   "global_embed_dim": 64, "global_ffn_mult": 2, "global_n_heads": 4}


def setup_runtime():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(name, overrides=None):
    import yaml
    if name not in POLICY_METHODS:
        raise ValueError(f"Unknown method: {name}")
    config_name = "refil" if name in V4_METHODS or name in V5_METHODS else name
    base = yaml.safe_load((UPSTREAM / "config/default.yaml").read_text(encoding="utf-8-sig"))
    base = _merge(base, yaml.safe_load((PROJECT / f"configs/{config_name}.yaml").read_text(encoding="utf-8-sig")))
    if VERSION == "main_v5" and name in V5_OVERRIDES:
        _merge(base, V5_OVERRIDES[name])
    elif name in V4_OVERRIDES:
        _merge(base, V4_OVERRIDES[name])
    base.update(method=name, name=name, env="had", seed=0, entity_scheme=True,
                max_traj_len=-1, popart=False, buffer_cpu_only=True,
                buffer_opt_mem=True, use_cuda=True, learner_log_interval=1000,
                mask_subtask_actions=False, feature_layout="had", report_interval=300,
                progress_interval=10, resume=False, run=FORMAL_RUN,
                # Potential-based shaping over Blue's closing distance. It only
                # retimes the signal: the discounted shaping sum telescopes to a
                # constant of the start state, so D and the anchors are unchanged.
                shaping_coef=1.0, shaping_range=4000.0)
    base["implementation_revision"] = "train_pool_pad_eval_ceiling"
    # ALMA is the only arm that drives the vendored hierarchy. COPA is not an
    # arm of this experiment, so the coach stays off for every method.
    base["hier_agent"]["copa"] = False
    base.setdefault("env_args", {})
    _merge(base, overrides or {})
    # The subtask keys exist exactly when the allocation layer is in use, and
    # the mixer follows the agent's conditioning as in the upstream run.py.
    base["multi_task"] = base["hier_agent"]["task_allocation"] is not None
    if not base["multi_task"] and base["agent"]["subtask_cond"] is not None:
        raise ValueError("Subtask conditioning requires an allocation layer")
    mixer_cond = base.get("mixer_subtask_cond")
    if mixer_cond in (False, "off", "none", "null"):
        base["mixer_subtask_cond"] = None
    elif mixer_cond is None:
        base["mixer_subtask_cond"] = base["agent"]["subtask_cond"]
    base["output"] = str(Path(base.get("output", DEFAULT_OUTPUT)).resolve())
    base["feature_layout"] = "had" if base["env"] == "had" else "native"
    base["device"] = "cuda" if base["use_cuda"] else "cpu"
    base.setdefault("imagine_group", "original")
    base.setdefault("count_cond", None)
    base.setdefault("count_ln", True)
    base.setdefault("global_branch", None)
    base.setdefault("global_slots", 4)
    base.setdefault("global_depths", [1, 2, 3, 4])
    base.setdefault("global_eval_depth", 4)
    base.setdefault("global_embed_dim", 64)
    base.setdefault("global_ffn_mult", 2)
    base.setdefault("global_n_heads", 4)
    base["global_depths"] = [int(value) for value in base["global_depths"]]
    base["global_slots"] = int(base["global_slots"])
    base["global_eval_depth"] = int(base["global_eval_depth"])
    base["global_embed_dim"] = int(base["global_embed_dim"])
    base["global_ffn_mult"] = int(base["global_ffn_mult"])
    base["global_n_heads"] = int(base["global_n_heads"])
    base.setdefault("reward_mode", "damage")
    base.setdefault("friendly_penalty", 1.0)
    base["friendly_penalty"] = float(base["friendly_penalty"])
    return SimpleNamespace(**base)


def make_runtime_env(args_dict, rank=0):
    setup_runtime()
    from open_score.envs.entity_env import HADEntityEnv, NativeEntityEnv
    env_args = dict(args_dict.get("env_args", {}))
    env_args["seed"] = int(args_dict["seed"]) + 1009 * int(rank)
    if args_dict["env"] == "had":
        # The folded Red-wipeout tail and the shaping term both use the
        # learner's own discount, and an ordered baseline refuses rosters its
        # slots cannot hold.
        env_args.setdefault("gamma", float(args_dict["gamma"]))
        for key in ("shaping_coef", "shaping_range"):
            if args_dict.get(key) is not None:
                env_args.setdefault(key, float(args_dict[key]))
        if args_dict.get("reward_mode") is not None:
            env_args.setdefault("reward_mode", args_dict["reward_mode"])
        if args_dict.get("friendly_penalty") is not None:
            env_args.setdefault("friendly_penalty", float(args_dict["friendly_penalty"]))
        if args_dict.get("pool_slots") is not None:
            env_args.setdefault("pool_slots", args_dict["pool_slots"])
        env_args.setdefault("pad", args_dict.get("entity_pad", "train"))
        hier = args_dict.get("hier_agent") or {}
        if hier.get("action_length") is not None:
            env_args.setdefault("action_length", int(hier["action_length"]))
        return HADEntityEnv(**env_args)
    return NativeEntityEnv(args_dict["env"], **env_args)


def make_scheme(env_info, multi_task=False):
    import torch as th
    from components.transforms import OneHot
    na, ne, nf, nu = (int(env_info[key]) for key in ("n_agents", "n_entities", "entity_shape", "n_actions"))
    state_shape = env_info.get("state_shape", ne * nf)
    scheme = {
        "entities": {"vshape": (ne, nf)},
        "obs_mask": {"vshape": (ne, ne), "dtype": th.uint8},
        "entity_mask": {"vshape": (ne,), "dtype": th.uint8},
        "agent_mask": {"vshape": (na,), "dtype": th.uint8},
        "initial_agent_mask": {"vshape": (na,), "dtype": th.uint8},
        "state": {"vshape": (int(state_shape),)},
        "avail_actions": {"vshape": (nu,), "group": "agents", "dtype": th.int32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reset": {"vshape": (1,), "dtype": th.uint8},
        "t_added": {"vshape": (1,), "dtype": th.long, "episode_const": True},
    }
    if multi_task:
        # ALMA's subtask contract: which entity belongs to which subtask, which
        # subtask slots are real, when the upper layer re-decides, and the
        # per-subtask reward/termination its low-level controllers learn from.
        nt = int(env_info["n_tasks"])
        scheme.update({
            "entity2task_mask": {"vshape": (ne, nt), "dtype": th.uint8},
            "task_mask": {"vshape": (nt,), "dtype": th.uint8},
            "hier_decision": {"vshape": (1,), "dtype": th.uint8},
            "task_rewards": {"vshape": (nt,)},
            "tasks_terminated": {"vshape": (nt,), "dtype": th.uint8},
        })
    return scheme, {"agents": na}, {"actions": ("actions_onehot", [OneHot(out_dim=nu)])}


class LearnerLogger:
    def __init__(self):
        self.console_logger = logging.getLogger("crossscale.learner")
        self.latest = {}

    def log_stat(self, name, value, t_env):
        self.latest[name] = float(value)


def build_learner(mac, scheme, logger, args):
    if args.method == "dcg":
        from .dcg_patch.dcg_learner import DCGLearner
        return DCGLearner(mac, scheme, logger, args)
    if args.method == "spectra":
        from .spectra_patch.nq_learner import NQLearner
        return NQLearner(mac, scheme, logger, args)
    from learners.q_learner import QLearner
    return QLearner(mac, scheme, logger, args)


def _network_state(learner):
    state = {"agent": learner.mac.agent.state_dict(), "target_agent": learner.target_mac.agent.state_dict(),
             "optimizer": learner.optimiser.state_dict(),
             "last_target_update_episode": learner.last_target_update_episode}
    if learner.mixer is not None:
        state.update(mixer=learner.mixer.state_dict(), target_mixer=learner.target_mixer.state_dict())
    if hasattr(learner.mac, "state_dict"):
        state["mac"] = learner.mac.state_dict()
        state["target_mac"] = learner.target_mac.state_dict()
    if hasattr(learner, "alloc_pi_optimiser"):
        # The allocation layer has its own two optimizers and target clock. A
        # checkpoint without them could not reproduce the policy it scored.
        state.update(alloc_pi_optimizer=learner.alloc_pi_optimiser.state_dict(),
                     alloc_q_optimizer=learner.alloc_q_optimiser.state_dict(),
                     last_alloc_target_update_episode=learner.last_alloc_target_update_episode)
    return state


def _load_network_state(learner, state):
    learner.mac.agent.load_state_dict(state["agent"])
    learner.target_mac.agent.load_state_dict(state["target_agent"])
    if "mac" in state:
        learner.mac.load_state_dict(state["mac"])
        learner.target_mac.load_state_dict(state["target_mac"])
    if hasattr(learner, "alloc_pi_optimiser"):
        learner.alloc_pi_optimiser.load_state_dict(state["alloc_pi_optimizer"])
        learner.alloc_q_optimiser.load_state_dict(state["alloc_q_optimizer"])
        learner.last_alloc_target_update_episode = state["last_alloc_target_update_episode"]
    if learner.mixer is not None:
        learner.mixer.load_state_dict(state["mixer"])
        learner.target_mixer.load_state_dict(state["target_mixer"])
    learner.optimiser.load_state_dict(state["optimizer"])
    learner.last_target_update_episode = state["last_target_update_episode"]


def _atomic_save(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(value, pending)
    os.replace(pending, path)


def _set_seed(seed, cuda):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def train(name, cfg):
    """Train one configured method; formal budgets must be explicit."""
    setup_runtime()
    import numpy as np
    import torch as th
    from components.episode_buffer import ReplayBuffer
    from runners.parallel_runner import ParallelRunner
    from open_score.models import build_mac
    from open_score.utils.logging import ExperimentLogger, read_records
    from open_score.eval.protocol import evaluation_thresholds, validation_jobs, validation_score, final_jobs, remaining_jobs
    from open_score.eval.report import refresh_report

    stop_event = cfg.get("_stop_event")
    cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
    if "t_max" not in cfg:
        raise ValueError("train requires an explicit t_max physical-step budget")
    args = load_config(name, cfg)
    if int(args.t_max) <= 0:
        raise ValueError("t_max must be an explicitly selected positive physical-step budget")
    if args.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    _set_seed(args.seed, args.device == "cuda")
    output = Path(args.output)
    run_dir = output / name / args.run / f"seed_{args.seed}"
    resume_path = run_dir / "resume.pt"
    if resume_path.exists() and not args.resume:
        raise FileExistsError(f"Existing run: {run_dir}; use --resume to continue it")
    if args.resume and getattr(args, "global_branch", None) not in (None, False, "off", "none", ""):
        saved_cfg_path = run_dir / "config.json"
        if saved_cfg_path.exists():
            saved_cfg = json.loads(saved_cfg_path.read_text(encoding="utf-8"))
            saved_arch = (
                int(saved_cfg["global_embed_dim"]) if "global_embed_dim" in saved_cfg else 128,
                int(saved_cfg["global_ffn_mult"]) if "global_ffn_mult" in saved_cfg else 4,
                int(saved_cfg["global_n_heads"]) if "global_n_heads" in saved_cfg else int(saved_cfg.get("attn_n_heads", 4)),
            )
            current_arch = (
                int(getattr(args, "global_embed_dim", 64)),
                int(getattr(args, "global_ffn_mult", 2)),
                int(getattr(args, "global_n_heads", 4)),
            )
            if saved_arch != current_arch:
                print(f"[{name}] global module {saved_arch} -> {current_arch}; "
                      "start this arm from step 0 (eval protocol unchanged)", flush=True)
                args.resume = False
    run_dir.mkdir(parents=True, exist_ok=True)
    record = ExperimentLogger(output, name, args.seed, args.run)
    logger = LearnerLogger()
    runner = None
    stop_requested = False
    old_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"[{name}] stop: finish batch and save resume", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    state = dict(episode_num=0, updates=0, next_eval=1, best_score=float("inf"),
                 elapsed_seconds=0.0, training_seconds=0.0, status="running")
    # psutil is already installed in the experiment environment; no new
    # monitoring process or service is started.
    import psutil
    process = psutil.Process()
    resource_previous = (time.monotonic(), 0.0)
    resource_latest = {}

    def sample_resources():
        nonlocal resource_previous, resource_latest
        rss = private = cpu_seconds = 0.0
        for member in [process, *process.children(recursive=True)]:
            try:
                memory = member.memory_info()
                rss += memory.rss
                private += getattr(memory, "private", memory.rss)
                cpu = member.cpu_times()
                cpu_seconds += cpu.user + cpu.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        now = time.monotonic()
        cpu_cores = max(0.0, cpu_seconds - resource_previous[1]) / max(now - resource_previous[0], 1e-6)
        resource_previous = (now, cpu_seconds)
        resource_latest = dict(process_tree_rss_gib=rss / 2**30,
                               process_tree_private_gib=private / 2**30,
                               cpu_cores=cpu_cores,
                               system_available_gib=psutil.virtual_memory().available / 2**30)
        if args.device == "cuda":
            resource_latest.update(cuda_allocated_gib=th.cuda.memory_allocated() / 2**30,
                                   cuda_peak_allocated_gib=th.cuda.max_memory_allocated() / 2**30,
                                   cuda_reserved_gib=th.cuda.memory_reserved() / 2**30)
        state["peak_private_gib"] = max(state.get("peak_private_gib", 0), private / 2**30)
        state["peak_rss_gib"] = max(state.get("peak_rss_gib", 0), rss / 2**30)
        state["peak_cuda_allocated_gib"] = max(state.get("peak_cuda_allocated_gib", 0), resource_latest.get("cuda_peak_allocated_gib", 0))
        state["measurement_peak_cuda_gib"] = max(state.get("measurement_peak_cuda_gib", 0), resource_latest.get("cuda_peak_allocated_gib", 0))
        state["measurement_peak_private_gib"] = max(state.get("measurement_peak_private_gib", 0), private / 2**30)
        state["measurement_peak_rss_gib"] = max(state.get("measurement_peak_rss_gib", 0), rss / 2**30)
        return resource_latest
    started = time.monotonic()
    last_progress = last_report = last_save = started
    saved_t_env = 0
    # The formal protocol scores HAD damage, so it cannot run on a sanity env.
    if args.run == FORMAL_RUN and args.env != "had":
        raise ValueError(f"The formal {FORMAL_RUN} run requires env=had, not {args.env!r};"
                         " pass an explicit --run for native sanity checks")
    formal = args.run == FORMAL_RUN
    eval_progress = dict(completed=0, total=0)
    args.entity_pad = "train"
    if args.resume and (run_dir / "config.json").exists():
        saved_n = int(json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("n_agents", 10))
        if saved_n > 10:
            args.entity_pad = "eval"
    try:
        runner = ParallelRunner(args, logger)
        env_info = runner.get_env_info()
        for key, value in env_info.items():
            setattr(args, key, value)
        scheme, groups, preprocess = make_scheme(env_info, multi_task=args.multi_task)
        # ALMA controllers expect the base feature dimension as a scalar.
        model_scheme = copy.deepcopy(scheme)
        model_scheme["entities"]["vshape"] = args.entity_shape
        model_scheme["actions_onehot"] = {"vshape": (args.n_actions,), "group": "agents"}
        mac = build_mac(model_scheme, groups, args)
        runner.setup(scheme, groups, preprocess, mac)
        learner = build_learner(mac, model_scheme, logger, args)
        if args.device == "cuda":
            learner.cuda()
        replay = ReplayBuffer(scheme, groups, args.buffer_size, args.episode_limit + 1,
                              preprocess=preprocess, device="cpu", efficient_store=True, max_traj_len=-1)
        if args.resume:
            if not resume_path.exists():
                raise FileNotFoundError(resume_path)
            saved = th.load(resume_path, map_location="cpu", weights_only=False)
            saved_config = saved["config"]
            current = vars(args)
            differing = sorted(key for key in set(saved_config) | set(current)
                               if key not in RESUME_MUTABLE
                               and saved_config.get(key, RESUME_DEFAULTS.get(key)) != current.get(key))
            if differing:
                raise ValueError(
                    "Resume configuration differs: " + ", ".join(differing)
                    + ". Old replay data may not be mixed with a changed experiment;"
                    " start a new --run instead.")
            _load_network_state(learner, saved["networks"])
            replay = saved["replay"]
            replay.to("cpu")
            runner.load_state_dict(saved["runner"])
            state.update(saved["progress"])
            random.setstate(saved["rng"]["python"])
            np.random.set_state(saved["rng"]["numpy"])
            th.set_rng_state(saved["rng"]["torch"].cpu())
            if args.device == "cuda":
                th.cuda.set_rng_state_all([item.cpu() for item in saved["rng"]["cuda"]])
            saved_t_env = runner.t_env
        if state.get("measurement_revision") != args.implementation_revision:
            state.update(measurement_revision=args.implementation_revision,
                         measurement_start_t_env=runner.t_env,
                         measurement_start_training_seconds=state["training_seconds"],
                         measurement_peak_cuda_gib=0.0, measurement_peak_private_gib=0.0,
                         measurement_peak_rss_gib=0.0)
        (run_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
        record.progress(status="resuming" if args.resume else "starting", phase=args.env,
                        t_env=runner.t_env, budget_steps=args.t_max, checkpoint_t_env=saved_t_env,
                        updates=state["updates"], elapsed_seconds=state["elapsed_seconds"])

        def save(kind, full=False):
            nonlocal saved_t_env, last_save
            value = dict(config=vars(args), env_info=env_info, networks=_network_state(learner),
                         progress={**state, "t_env": runner.t_env,
                                   "elapsed_seconds": state["elapsed_seconds"] + time.monotonic() - started})
            if full:
                value.update(replay=replay, runner=runner.state_dict(),
                             rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                      torch=th.get_rng_state(),
                                      cuda=th.cuda.get_rng_state_all() if args.device == "cuda" else []))
            path = run_dir / f"{kind}.pt"
            _atomic_save(value, path)
            if full:
                saved_t_env = runner.t_env
                last_save = time.monotonic()
            return path

        def heartbeat(evaluating=False, done=0, total=0, in_flight=0):
            nonlocal last_progress, stop_requested
            if (stop_event is not None and stop_event.is_set()) or (output / "stop.request").exists():
                stop_requested = True
            now = time.monotonic()
            if now - last_progress < args.progress_interval:
                return
            elapsed = state["elapsed_seconds"] + now - started
            metrics = dict(status="evaluating" if evaluating else "training", phase=args.env,
                           t_env=runner.t_env, budget_steps=args.t_max,
                           steps_per_second=runner.t_env / max(elapsed, 1e-6),
                           eval_completed=eval_progress["completed"] + done if evaluating else 0,
                           eval_total=eval_progress["total"] if evaluating else 0,
                           latest_validation_D=state.get("latest_validation_D"),
                           checkpoint_t_env=saved_t_env,
                           updates=state["updates"], elapsed_seconds=elapsed,
                           loss=getattr(learner, "last_metrics", {}).get("loss"))
            metrics.update(sample_resources())
            record.progress(**metrics)
            loss_label = "-" if metrics["loss"] is None else f"{metrics['loss']:.4f}"
            d_label = "-" if state.get("latest_validation_D") is None else f"{state['latest_validation_D']:.3f}"
            elapsed_h = int(elapsed // 3600)
            elapsed_m = int((elapsed % 3600) // 60)
            elapsed_s = int(elapsed % 60)
            print(f"[{name}] {elapsed_h}:{elapsed_m:02d}:{elapsed_s:02d}  "
                  f"{runner.t_env:,}/{args.t_max:,}  {metrics['steps_per_second']:.1f}/s  "
                  f"upd={state['updates']}  L={loss_label}"
                  + (f"  ev={metrics['eval_completed']}/{metrics['eval_total']}" if evaluating else "")
                  + f"  D={d_label}", flush=True)
            last_progress = now

        runner.progress_callback = heartbeat

        def values(batch, t, actions, active):
            # Use the outputs already computed by select_actions: do not advance a GRU twice.
            if hasattr(mac, "evaluation_values"):
                return mac.evaluation_values(batch, t, actions, active, learner.mixer)
            return {"q_tot": [None] * len(active), "q_i": [None] * len(active)}

        runner.value_callback = values

        def evaluate_jobs(jobs):
            # Validation must not consume the future training random streams.
            rng = (random.getstate(), np.random.get_state(), th.get_rng_state(),
                   th.cuda.get_rng_state_all() if args.device == "cuda" else [])
            runner_state = runner.state_dict()
            previous = read_records(output, "episodes", run=args.run, method=name, seed=args.seed)
            pending = remaining_jobs(jobs, previous)
            eval_progress.update(completed=len(jobs) - len(pending), total=len(jobs))
            fresh = []
            try:
                for offset in range(0, len(pending), args.batch_size_run):
                    eval_progress["completed"] = len(jobs) - len(pending) + offset
                    group = pending[offset:offset + args.batch_size_run]
                    _, summaries = runner.run(test_mode=True, jobs=group)
                    rows = [{**summary, **job, "train_dist": "mixed_le10"}
                            for summary, job in zip(summaries, group)]
                    for job, trajectory in zip(group, runner.last_trajectories):
                        if job.get("retain_trajectory") and trajectory is not None:
                            record.trajectories([{**job, "trajectory": trajectory}])
                    record.episodes(rows)
                    fresh.extend(rows)
                    if stop_requested:
                        break
                return previous + fresh
            finally:
                random.setstate(rng[0]); np.random.set_state(rng[1]); th.set_rng_state(rng[2])
                if args.device == "cuda":
                    th.cuda.set_rng_state_all(rng[3])
                runner.load_state_dict(runner_state)

        thresholds = evaluation_thresholds(args.t_max) if formal else []
        def run_pending_evaluations():
            while formal and state["next_eval"] <= 50 and runner.t_env >= thresholds[state["next_eval"] - 1]:
                # Save this exact evaluation policy before starting a resumable validation point.
                save("resume", full=True)
                point = state["next_eval"]
                rows = evaluate_jobs(validation_jobs(point, runner.t_env))
                current = [row for row in rows if row.get("phase") == "train_eval" and row.get("eval_point") == point]
                score = validation_score(current)
                if score is None:
                    if stop_requested:
                        break
                    raise RuntimeError(f"Validation point {point} is incomplete")
                state["latest_validation_D"] = score
                if score < state["best_score"]:
                    # Commit the weights first: a failed write must not leave a
                    # recorded score that no retained model can reproduce.
                    save("best")
                    state["best_score"] = score
                    state["best_t_env"] = runner.t_env
                state["next_eval"] += 1
                refresh_report(output)
                if stop_requested:
                    break

        run_pending_evaluations()
        while runner.t_env < args.t_max and not stop_requested:
            cycle_start = time.monotonic()
            episode_batch, summaries = runner.run(max_train_steps=args.t_max - runner.t_env)
            collected_at = time.monotonic()
            episode_batch.to("cpu")
            replay.insert_episode_batch(episode_batch)
            state["episode_num"] += episode_batch.batch_size
            update_metrics = []
            if replay.can_sample(args.batch_size):
                for _ in range(args.training_iters):
                    sample = replay.sample(args.batch_size)
                    max_t = int(sample.max_t_filled().item())
                    sample = sample[:, :max_t]
                    sample.to(args.device)
                    metrics = learner.train(sample, runner.t_env, state["episode_num"])
                    state["updates"] += 1
                    if isinstance(metrics, dict):
                        update_metrics.append(metrics)
                    if args.hier_agent["task_allocation"] == "aql":
                        # ALMA trains the allocation layer on its own draw, so
                        # its Q-loss can drop episodes older than decay_old.
                        filters = {}
                        if args.hier_agent["decay_old"] > 0:
                            cutoff = int(args.hier_agent["decay_old"])
                            filters["t_added"] = lambda added: (runner.t_env - added) <= cutoff
                        if replay.can_sample(args.batch_size, filters=filters):
                            alloc_sample = replay.sample(args.batch_size, filters=filters)
                            alloc_sample = alloc_sample[:, :int(alloc_sample.max_t_filled().item())]
                            alloc_sample.to(args.device)
                            learner.alloc_train_aql(alloc_sample, runner.t_env, state["episode_num"])
            state["training_seconds"] += time.monotonic() - cycle_start
            metrics = dict(logger.latest)
            if update_metrics:
                for key in update_metrics[-1]:
                    metrics[key] = float(np.mean([item[key] for item in update_metrics if key in item]))
            metrics.update(updates=state["updates"], return_mean=float(np.mean([s["return"] for s in summaries])),
                           implementation_revision=args.implementation_revision,
                           ep_len_mean=float(np.mean([s["ep_len"] for s in summaries])),
                           collect_seconds=collected_at - cycle_start,
                           learn_seconds=time.monotonic() - collected_at,
                           collect_physical_steps=runner.env_steps_this_run,
                           training_seconds=state["training_seconds"])
            if args.env == "had":
                metrics["train_D"] = float(np.mean([s["D"] for s in summaries]))
                metrics["train_rho"] = float(np.mean([s["rho"] for s in summaries]))
                # Length of the Red-wipeout tail that is folded into one
                # terminal reward instead of entering the replay.
                metrics["train_wipeout_steps"] = float(np.mean([s["wipeout_steps"] for s in summaries]))
            record.learning(runner.t_env, metrics)
            run_pending_evaluations()
            heartbeat()
            if time.monotonic() - last_save >= args.report_interval:
                save("resume", full=True)
            if time.monotonic() - last_report >= args.report_interval:
                refresh_report(output)
                last_report = time.monotonic()

        state["status"] = "stopped" if stop_requested else "completed"
        if not stop_requested:
            save("final")
            if formal:
                from open_score.envs.features import MAX_AGENTS
                best_path = run_dir / "best.pt"
                saved = th.load(best_path, map_location="cpu", weights_only=False)
                if int(args.n_agents) >= MAX_AGENTS:
                    final_state = copy.deepcopy(_network_state(learner))
                    _load_network_state(learner, saved["networks"])
                    try:
                        evaluate_jobs(final_jobs(int(saved["progress"]["t_env"]), method=name))
                    finally:
                        _load_network_state(learner, final_state)
                else:
                    from open_score.eval.protocol import evaluate_checkpoint
                    evaluate_checkpoint(name, best_path, output=output, run=args.run,
                                        seed=args.seed, stop_requested=lambda: stop_requested)
                if stop_requested:
                    state["status"] = "stopped"
        path = save("resume", full=True)
        elapsed = state["elapsed_seconds"] + time.monotonic() - started
        resources = sample_resources()
        record.progress(status=state["status"], phase=args.env, t_env=runner.t_env,
                        budget_steps=args.t_max, elapsed_seconds=elapsed, updates=state["updates"],
                        steps_per_second=runner.t_env / max(elapsed, 1e-6), checkpoint_t_env=saved_t_env,
                        **resources)
        if args.run.startswith("benchmark"):
            record.benchmarks([dict(batch_size_run=args.batch_size_run,
                                   concurrency=getattr(args, "concurrency", 1), t_env=runner.t_env,
                                   wall_seconds=elapsed, training_seconds=state["training_seconds"],
                                   steps_per_second=runner.t_env / max(state["training_seconds"], 1e-6),
                                   peak_private_gib=state["peak_private_gib"],
                                   peak_rss_gib=state["peak_rss_gib"],
                                   peak_cuda_allocated_gib=state["peak_cuda_allocated_gib"],
                                   implementation_revision=args.implementation_revision,
                                   measurement_start_t_env=state["measurement_start_t_env"],
                                   measurement_steps=runner.t_env - state["measurement_start_t_env"],
                                   measurement_seconds=state["training_seconds"] - state["measurement_start_training_seconds"],
                                   current_steps_per_second=(runner.t_env - state["measurement_start_t_env"]) /
                                       max(state["training_seconds"] - state["measurement_start_training_seconds"], 1e-6),
                                   current_peak_cuda_gib=state["measurement_peak_cuda_gib"],
                                   current_peak_private_gib=state["measurement_peak_private_gib"],
                                   current_peak_rss_gib=state["measurement_peak_rss_gib"],
                                   status=state["status"])])
        refresh_report(output)
        return str(path if stop_requested else run_dir / "final.pt")
    except BaseException as error:
        if runner is not None and "save" in locals():
            try:
                save("resume", full=True)
            except BaseException:
                pass
        record.progress(status="failed", error=f"{type(error).__name__}: {error}",
                        t_env=runner.t_env if runner else 0, checkpoint_t_env=saved_t_env)
        refresh_report(output)
        raise
    finally:
        if runner is not None:
            runner.close_env()
        signal.signal(signal.SIGINT, old_sigint)


def load_policy(name, checkpoint):
    setup_runtime()
    from open_score.models import build_mac
    import torch as th
    saved = th.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["config"]["method"] != name:
        raise ValueError("Checkpoint method does not match requested policy")
    args = SimpleNamespace(**saved["config"])
    args.device = "cpu"
    # Set-based checkpoints do not bake roster size into weights. Rebuild the
    # pad to the current entity interface so 40v40 / 20v40 can be evaluated.
    from open_score.envs.entity_env import HADEntityEnv
    env_args = dict(getattr(args, "env_args", None) or {})
    probe_kwargs = {}
    if env_args.get("subtask_set"):
        probe_kwargs["subtask_set"] = env_args["subtask_set"]
    probe = HADEntityEnv(seed=0, scale=(4, 4, 2), **probe_kwargs)
    env_info = probe.get_env_info()
    probe.close()
    if getattr(args, "multi_task", False):
        # The subtask one-hot width is baked into the task embeddings, so it
        # stays fixed while n_tasks grows from the training pad to the
        # evaluation interface; the spare embeddings absorb the difference.
        width = int(args.n_tasks) + int(args.n_extra_tasks)
        args.n_extra_tasks = width - int(env_info["n_tasks"])
        if args.n_extra_tasks < 0:
            raise ValueError(f"checkpoint carries {width} subtask embeddings but the evaluation "
                             f"interface has {env_info['n_tasks']}; train with a larger n_extra_tasks")
    for key in ("n_agents", "n_entities", "entity_shape", "state_shape", "obs_shape", "n_tasks"):
        if key in env_info:
            setattr(args, key, env_info[key])
    scheme, groups, preprocess = make_scheme(env_info, multi_task=getattr(args, "multi_task", False))
    model_scheme = copy.deepcopy(scheme)
    model_scheme["entities"]["vshape"] = args.entity_shape
    model_scheme["actions_onehot"] = {"vshape": (args.n_actions,), "group": "agents"}
    mac = build_mac(model_scheme, groups, args)
    mac.agent.load_state_dict(saved["networks"]["agent"])
    if "mac" in saved["networks"]:
        mac.load_state_dict(saved["networks"]["mac"])
    from open_score.envs.entity_env import FrozenPolicyAdapter
    from open_score.models import build_mixer
    mixer = build_mixer(args)
    if mixer is not None:
        mixer.load_state_dict(saved["networks"]["mixer"])
        mixer.eval()
    return FrozenPolicyAdapter(mac, args, scheme, groups, preprocess, mixer)


__all__ = ["train", "load_policy", "load_config", "METHODS", "PROBE_METHODS",
           "V4_METHODS", "V5_METHODS", "V5_GLOBAL_METHODS", "POLICY_METHODS"]
