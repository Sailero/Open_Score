"""Shared ALMA training entry and frozen-policy interface for HAD experiments."""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import random
import re
import signal
import sys
import time
import traceback
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
_CYCLE = {
    "imagine_group": "original",
    "count_cond": "phi2",
    "global_branch": "cycle",
    "global_slots": 4,
    "global_depths": [1, 2, 3, 4],
    "global_eval_depth": 4,
    "global_embed_dim": 64,
    "global_ffn_mult": 2,
    "global_n_heads": 4,
}
MAIN_METHODS = ("regir", "refil", "b2_qmix_atten", "dcg", "spectra", "alma")
MAIN_ABLATION_METHODS = ("regir_norefil", "regir_nocount", "regir_r1", "regir_last")
MAIN0921_METHODS = ("transfqmix", "regir_fixed4", "regir_untied4", "regir_kv0")
MAIN_TRAIN_METHODS = MAIN_METHODS + MAIN_ABLATION_METHODS + ("refil_matched",) + MAIN0921_METHODS
MAIN0923_METHODS = ("regir_r0", "regir_kv0_norefil", "regir_kv0_fixed4", "regir_kv0_nomem",
                    "regir_kv0_nocount", "regir_prenorm",
                    "regir_r1_sg", "regir_kv0_sg", "regir_sg", "regir_untied4_sg", "regir_fixed4_sg",
                    "regir_r0_sg", "regir_kv0_norefil_sg", "regir_kv0_fixed4_sg", "regir_kv0_nocount_sg",
                    "regir_kv0_intent_sg", "regir_kv0_intent_noaux_sg", "regir_kv0_intent_nomem",
                    "regir_intent_sg", "regir_kv0_intent_norefil_sg",
                    "regir_r1_nomem", "regir_r1_norefil_sg", "regir_r1_nocount_sg")
METHOD_ALIASES = {"refil_cycle": "regir", "regia": "regir"}
_NOREFIL = {"lmbda": 0.0, "skip_refil_local": True, "agent": {"imagine": False}}
_INTENT = {"rer_intent": True, "intent_aux_weight": 0.02}


def _sg(cfg):
    return {**cfg, "global_query_detach_memory": True}
MAIN_OVERRIDES = {
    "regir": dict(_CYCLE),
    "refil_cycle": dict(_CYCLE),
    "regir_norefil": {**_CYCLE, "lmbda": 0.0, "skip_refil_local": True,
                      "agent": {"imagine": False}},
    "regir_nocount": {**_CYCLE, "skip_count_inject": True},
    "regir_r1": {**_CYCLE, "global_depths": [1], "global_eval_depth": 1},
    "regir_last": {**_CYCLE, "read_last_round": True},
    "regir_fixed4": {**_CYCLE, "global_depths": [4]},
    "regir_untied4": {**_CYCLE, "global_depths": [4], "rer_update": "untied4"},
    "regir_kv0": {**_CYCLE, "rer_update": "kv0"},
    "regir_r0": {**_CYCLE, "global_depths": [1], "global_eval_depth": 1, "global_read_h0": True},
    "regir_kv0_norefil": {**_CYCLE, "rer_update": "kv0", "lmbda": 0.0, "skip_refil_local": True,
                          "agent": {"imagine": False}},
    "regir_kv0_fixed4": {**_CYCLE, "rer_update": "kv0", "global_depths": [4]},
    "regir_kv0_nomem": {**_CYCLE, "rer_update": "kv0", "global_query_no_memory": True},
    "regir_kv0_nocount": {**_CYCLE, "rer_update": "kv0", "skip_count_inject": True},
    "regir_prenorm": {**_CYCLE, "global_kv_prenorm": True},
    "transfqmix": {"mac": "transfqmix_mac", "learner": "transfqmix_learner",
                    "mixer": "transfqmix", "lr": .001, "weight_decay": 0,
                    "optimizer": "adam", "gamma": .99, "td_lambda": .6,
                    "grad_norm_clip": 10, "buffer_size": 5000, "batch_size": 32,
                    "training_iters": 1,
                    "epsilon_start": 1., "epsilon_finish": .05, "epsilon_anneal_time": 100000,
                    "target_update_interval": 200, "entity_last_action": False,
                    "obs_agent_id": False, "obs_last_action": False,
                    "emb": 32, "heads": 4, "depth": 2, "mixer_emb": 32,
                    "mixer_heads": 4, "mixer_depth": 2, "ff_hidden_mult": 4,
                    "dropout": 0., "agent": {"imagine": False}, "lmbda": 0.},
    "refil_matched": {"imagine_group": "original", "global_branch": None},
}
for _name in ("regir_r1", "regir_kv0", "regir", "regir_untied4", "regir_fixed4", "regir_r0",
              "regir_kv0_norefil", "regir_kv0_fixed4", "regir_kv0_nocount"):
    MAIN_OVERRIDES[f"{_name}_sg"] = _sg(MAIN_OVERRIDES[_name])
MAIN_OVERRIDES.update({
    "regir_kv0_intent_sg": _sg({**MAIN_OVERRIDES["regir_kv0"], **_INTENT}),
    "regir_kv0_intent_noaux_sg": _sg({**MAIN_OVERRIDES["regir_kv0"], **_INTENT, "intent_aux_weight": 0.0}),
    "regir_kv0_intent_nomem": {**MAIN_OVERRIDES["regir_kv0"], **_INTENT, "global_query_no_memory": True},
    "regir_intent_sg": _sg({**MAIN_OVERRIDES["regir"], **_INTENT}),
    "regir_kv0_intent_norefil_sg": _sg({**MAIN_OVERRIDES["regir_kv0"], **_INTENT, **_NOREFIL}),
    "regir_r1_nomem": {**MAIN_OVERRIDES["regir_r1"], "global_query_no_memory": True},
    "regir_r1_norefil_sg": _sg({**MAIN_OVERRIDES["regir_r1"], **_NOREFIL}),
    "regir_r1_nocount_sg": _sg({**MAIN_OVERRIDES["regir_r1"], "skip_count_inject": True}),
})
POLICY_METHODS = METHODS + PROBE_METHODS + tuple(dict.fromkeys(
    (*V4_METHODS, *V5_METHODS, *MAIN_TRAIN_METHODS, *MAIN0923_METHODS, "refil_cycle", "alma_legacy",
     "refil_count_ln")))
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
        "global_feedback_mode": "cycle_jk",
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
        "global_slots": 10,
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
                            "report_interval", "progress_interval", "resume_interval",
                            "implementation_revision", "concurrency",
                            "entity_pad", "skip_final_eval"))
# Keys added after some runs started; missing saved values equal these defaults.
RESUME_DEFAULTS = {"reward_mode": "damage", "friendly_penalty": 1.0,
                   "count_ln": True, "global_branch": None, "global_slots": 4,
                   "global_depths": [1, 2, 3, 4], "global_eval_depth": 4,
                   "global_embed_dim": 64, "global_ffn_mult": 2, "global_n_heads": 4,
                   "global_feedback_mode": "gru", "resume_interval": 600,
                   "skip_refil_local": False, "skip_count_inject": False,
                   "read_last_round": False, "matched_hidden": 0,
                   "skip_final_eval": False, "rer_update": "tied", "rer_readout": "learned"}
RESUME_NAMES = ("resume.pt", "resume.pt.pending", "resume.prev.pt")
DEFAULT_MATCHED_HIDDEN = 576


def resolve_matched_hidden(output):
    path = Path(output) / "matched_hidden.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8"))["matched_hidden"])
    return DEFAULT_MATCHED_HIDDEN


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


def canonical_method(name):
    return METHOD_ALIASES.get(name, name)


def _config_yaml_name(name):
    if name in ("refil_matched", "transfqmix") or name in V4_METHODS or name in V5_METHODS:
        return "refil"
    if name == "alma_legacy":
        return "alma"
    if canonical_method(name).startswith("regir"):
        return "refil"
    return name


def load_config(name, overrides=None):
    import yaml
    if name not in POLICY_METHODS:
        raise ValueError(f"Unknown method: {name}")
    config_name = _config_yaml_name(name)
    base = yaml.safe_load((UPSTREAM / "config/default.yaml").read_text(encoding="utf-8-sig"))
    base = _merge(base, yaml.safe_load((PROJECT / f"configs/{config_name}.yaml").read_text(encoding="utf-8-sig")))
    if name in MAIN_OVERRIDES:
        _merge(base, MAIN_OVERRIDES[name])
    elif name in V5_OVERRIDES:
        _merge(base, V5_OVERRIDES[name])
    elif name in V4_OVERRIDES:
        _merge(base, V4_OVERRIDES[name])
    if name == "alma_legacy":
        base["n_extra_tasks"] = 3
    base.update(method=name, name=name, env="had", seed=0, entity_scheme=True,
                max_traj_len=-1, popart=False, buffer_cpu_only=True,
                buffer_opt_mem=True, use_cuda=True, learner_log_interval=1000,
                mask_subtask_actions=False, feature_layout="had", report_interval=300,
                progress_interval=10, resume_interval=600, resume=False, run=FORMAL_RUN,
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
    if name == "refil_matched" and int(base.get("matched_hidden") or 0) <= 0:
        base["matched_hidden"] = resolve_matched_hidden(base["output"])
    base["feature_layout"] = ("smacv2" if base["env"] == "smacv2" else
                              "had" if base["env"] == "had" else "native")
    if base["env"] == "smacv2":
        base.update(entity_last_action=False, obs_last_action=False, obs_agent_id=False,
                    state_last_action=False, actor_entity_shape=32, observer_entity_shape=32,
                    action_head="common6_enemy")
        base["env_args"].update(obs_last_action=False, state_last_action=False)
        if name != "transfqmix":
            base["training_iters"] = 4  # Four completed collection episodes, as in HAD's 8/8 ratio.
        if name == "spectra":
            # Official SMACv2 ss_qmix.yaml; the HAD/GRF batch remains 32.
            base["batch_size"] = 128
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
    base.setdefault("global_feedback_mode", "gru")
    base.setdefault("skip_refil_local", False)
    base.setdefault("skip_count_inject", False)
    base.setdefault("read_last_round", False)
    base.setdefault("rer_update", "tied")
    base.setdefault("rer_readout", "learned")
    base.setdefault("matched_hidden", 0)
    base["global_depths"] = [int(value) for value in base["global_depths"]]
    base["global_slots"] = int(base["global_slots"])
    base["global_eval_depth"] = int(base["global_eval_depth"])
    base["global_embed_dim"] = int(base["global_embed_dim"])
    base["global_ffn_mult"] = int(base["global_ffn_mult"])
    base["global_n_heads"] = int(base["global_n_heads"])
    base["skip_refil_local"] = bool(base["skip_refil_local"])
    base["skip_count_inject"] = bool(base["skip_count_inject"])
    base["read_last_round"] = bool(base["read_last_round"])
    base["matched_hidden"] = int(base["matched_hidden"])
    base.setdefault("reward_mode", "damage")
    base.setdefault("friendly_penalty", 1.0)
    base["friendly_penalty"] = float(base["friendly_penalty"])
    base.setdefault("resume_interval", 600)
    base["resume_interval"] = int(base["resume_interval"])
    base.setdefault("skip_final_eval", False)
    base["skip_final_eval"] = bool(base["skip_final_eval"])
    return SimpleNamespace(**base)


def make_runtime_env(args_dict, rank=0):
    setup_runtime()
    from open_score.envs.entity_env import HADEntityEnv, NativeEntityEnv
    env_args = dict(args_dict.get("env_args", {}))
    env_args["seed"] = int(args_dict["seed"]) + 1009 * int(rank)
    if args_dict["env"] == "smacv2":
        from open_score.envs.smacv2_env import MixedScaleSMACAdapter
        env_args.setdefault("pad", args_dict.get("entity_pad", "train"))
        return MixedScaleSMACAdapter(**env_args)
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
    actor_width = env_info.get("actor_entity_shape", env_info.get("observer_entity_shape"))
    if actor_width is not None:
        scheme["observer_entities"] = {"vshape": (na, ne, int(actor_width))}
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
    if args.method == "transfqmix":
        from .transfqmix import TransfQMixLearner
        return TransfQMixLearner(mac, scheme, logger, args)
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


def resume_files(run_dir):
    run_dir = Path(run_dir)
    return tuple(run_dir / name for name in RESUME_NAMES)


def _fsync_file(path):
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_save(value, path, *, keep_previous=False):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(value, pending)
    _fsync_file(pending)
    if keep_previous and path.exists():
        os.replace(path, path.with_name(path.stem + ".prev" + path.suffix))
    os.replace(pending, path)
    _fsync_dir(path.parent)


def _load_resume(run_dir, *, require_complete=False):
    import torch
    candidates = [path for path in resume_files(run_dir) if path.exists()]
    if not candidates:
        return None, None
    errors = []
    for path in candidates:
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        if not isinstance(saved, dict) or any(key not in saved for key in ("replay", "runner", "rng", "networks")):
            errors.append(f"{path.name}: incomplete checkpoint")
            continue
        if require_complete and (saved.get("transaction_complete") is False or
                (saved.get("transaction_complete") is not True and
                 saved.get("progress", {}).get("status") == "failed")):
            errors.append(f"{path.name}: failed checkpoint has no complete-batch boundary")
            continue
        return saved, path
    raise RuntimeError("Resume files exist but none are readable: " + "; ".join(errors))


def _set_seed(seed, cuda):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def _physical_gpu():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return int(visible) if visible.isdigit() else None


def _cuda_oom_details(error):
    """Classify CUDA allocation failures without allocating another tensor."""
    from open_score.utils.resources import is_cuda_oom
    message = str(error)
    if not is_cuda_oom(error):
        return None
    def gib(pattern):
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match is None:
            return None
        unit = match.group(2).lower()
        factor = {"bytes": 1., "b": 1., "kib": 2.**10, "mib": 2.**20,
                  "gib": 2.**30, "tib": 2.**40, "kb": 1e3, "mb": 1e6,
                  "gb": 1e9, "tb": 1e12}[unit]
        return float(match.group(1)) * factor / 2.**30
    units = r"(bytes|[kmgt]i?b|b)"
    requested = gib(r"tried to allocate\s+([\d.]+)\s*" + units)
    free = gib(r"of which\s+([\d.]+)\s*" + units + r"\s+is free")
    return dict(requested_allocation_gib=requested,
                minimum_additional_gib=None if requested is None or free is None else max(0., requested - free))


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

    def refresh_report_safe():
        if profile:
            return
        try:
            refresh_report(output)
        except (OSError, MemoryError, TimeoutError) as error:
            print(f"[{name}] report refresh skipped: {type(error).__name__}", flush=True)

    stop_event = cfg.get("_stop_event")
    cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
    if "t_max" not in cfg:
        raise ValueError("train requires an explicit t_max physical-step budget")
    args = load_config(name, cfg)
    from open_score.eval import experiment
    profile = getattr(args, "profile", None) == experiment.PROFILE
    if profile:
        if (name not in experiment.methods(args.env)
                or int(args.seed) not in experiment.method_seeds(args.env, name)):
            raise ValueError(f"Method/environment/seed is outside the frozen {experiment.PROFILE} matrix")
        if args.run != "train" or int(args.t_max) != experiment.budget(args.env):
            raise ValueError(f"{experiment.PROFILE} requires run=train and its frozen environment budget")
        args.skip_final_eval = True
        args.implementation_revision = experiment.IMPLEMENTATION_REVISION
    if int(args.t_max) <= 0:
        raise ValueError("t_max must be an explicitly selected positive physical-step budget")
    output = Path(args.output)
    run_dir = (experiment.run_directory(output, name, args.seed, env=args.env, run=args.run) if profile
               else output / name / args.run / f"seed_{args.seed}")
    if any(path.exists() for path in resume_files(run_dir)) and not args.resume:
        raise FileExistsError(f"Existing run: {run_dir}; use --resume to continue it")
    if profile and not args.resume and any((run_dir / filename).exists()
                                           for filename in ("config.json", "final.pt", "best.pt")):
        raise FileExistsError(f"Existing {experiment.PROFILE} run must not be overwritten: {run_dir}")
    if args.resume and getattr(args, "global_branch", None) not in (None, False, "off", "none", ""):
        saved_cfg_path = run_dir / "config.json"
        if saved_cfg_path.exists():
            saved_cfg = json.loads(saved_cfg_path.read_text(encoding="utf-8"))
            saved_arch = (
                int(saved_cfg["global_embed_dim"]) if "global_embed_dim" in saved_cfg else 128,
                int(saved_cfg["global_ffn_mult"]) if "global_ffn_mult" in saved_cfg else 4,
                int(saved_cfg["global_n_heads"]) if "global_n_heads" in saved_cfg else int(saved_cfg.get("attn_n_heads", 4)),
                int(saved_cfg["global_slots"]) if "global_slots" in saved_cfg else 4,
                saved_cfg.get("global_feedback_mode", "gru") if saved_cfg.get("global_branch") == "feedback" else "gru",
            )
            current_arch = (
                int(getattr(args, "global_embed_dim", 64)),
                int(getattr(args, "global_ffn_mult", 2)),
                int(getattr(args, "global_n_heads", 4)),
                int(getattr(args, "global_slots", 4)),
                getattr(args, "global_feedback_mode", "gru") if getattr(args, "global_branch", None) == "feedback" else "gru",
            )
            if saved_arch != current_arch:
                raise ValueError(f"Resume global architecture differs: {saved_arch} -> {current_arch}; "
                                 "existing checkpoints and replay are preserved")
    run_dir.mkdir(parents=True, exist_ok=True)
    record = ExperimentLogger(output, name, args.seed, args.run, env=args.env)
    logger = LearnerLogger()
    runner = None
    stop_requested = False
    old_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"[{name}] stop: finish batch and save resume", flush=True)

    for sig in old_signals:
        signal.signal(sig, request_stop)
    state = dict(episode_num=0, updates=0, next_eval=1,
                 best_score=-float("inf") if profile and args.env == "smacv2" else float("inf"),
                 elapsed_seconds=0.0, training_seconds=0.0, validation_seconds=0.0,
                 validation_completed_episodes=0, validation_measured_episodes=0,
                 validation_measured_seconds=0.0, status="running")
    # psutil is already installed in the experiment environment; no new
    # monitoring process or service is started.
    import psutil
    process = psutil.Process()
    resource_previous = (time.monotonic(), 0.0)
    resource_latest = {}
    learner_ref = {}
    observed_update = False
    transaction_complete = True
    transaction_phase = "initializing"
    collection_steps_seen = 0
    safe_checkpoint_path = None
    safe_checkpoint_t_env = 0
    session_complete_batches = 0
    validation_clock = None

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
                                   cuda_reserved_gib=th.cuda.memory_reserved() / 2**30,
                                   cuda_peak_reserved_gib=th.cuda.max_memory_reserved() / 2**30)
        state["peak_private_gib"] = max(state.get("peak_private_gib", 0), private / 2**30)
        state["peak_rss_gib"] = max(state.get("peak_rss_gib", 0), rss / 2**30)
        state["peak_cuda_allocated_gib"] = max(state.get("peak_cuda_allocated_gib", 0), resource_latest.get("cuda_peak_allocated_gib", 0))
        state["measurement_peak_cuda_gib"] = max(state.get("measurement_peak_cuda_gib", 0), resource_latest.get("cuda_peak_allocated_gib", 0))
        state["measurement_peak_private_gib"] = max(state.get("measurement_peak_private_gib", 0), private / 2**30)
        state["measurement_peak_rss_gib"] = max(state.get("measurement_peak_rss_gib", 0), rss / 2**30)
        if profile:
            experiment.atomic_json(run_dir / "resource.json", dict(
                env=args.env, method=name, seed=int(args.seed),
                t_env=int(runner.t_env) if runner is not None else 0, budget_steps=int(args.t_max),
                cuda_peak_reserved_gib=resource_latest.get("cuda_peak_reserved_gib", 0.),
                cuda_reserved_gib=resource_latest.get("cuda_reserved_gib", 0.),
                status=state["status"], pid=os.getpid(), updated_at=time.time(), training_updates=int(state["updates"]),
                physical_gpu=_physical_gpu(), safe_checkpoint_t_env=int(safe_checkpoint_t_env),
                current_phase=transaction_phase, training_seconds=state["training_seconds"],
                training_steps_per_second=(runner.t_env / state["training_seconds"]
                    if runner is not None and state["training_seconds"] > 0 else None),
                # Total elapsed validation includes the current unfinished
                # group; throughput uses only the paired committed counters.
                validation_seconds=state["validation_seconds"] + (
                    now - validation_clock if validation_clock is not None else 0.0),
                validation_completed_episodes=int(state["validation_completed_episodes"]),
                validation_measured_episodes=int(state["validation_measured_episodes"]),
                validation_measured_seconds=state["validation_measured_seconds"],
                validation_total_episodes=(experiment.validation_point_count(args.env)
                    * (100 if args.env == "had" else 128)),
                eval_completed=eval_progress["completed"], eval_total=eval_progress["total"],
                grad_spikes=int(state.get("grad_spikes", 0)),
                grad_norm_last=getattr(learner_ref.get("learner"), "last_metrics", {}).get("grad_norm"),
                intent_loss=getattr(learner_ref.get("learner"), "last_metrics", {}).get("intent_loss"),
                intent_acc=getattr(learner_ref.get("learner"), "last_metrics", {}).get("intent_acc"),
                latest_validation_D=state.get("latest_validation_D"),
                estimate_ready=bool(observed_update)))
        return resource_latest
    started = time.monotonic()
    last_progress = last_report = last_save = started
    saved_t_env = 0
    # The formal protocol scores HAD damage, so it cannot run on a sanity env.
    if not profile and args.run == FORMAL_RUN and args.env != "had":
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
        # Discover and validate the retained CPU snapshot before touching GPU
        # model storage, so an initialization OOM can still name its safe source.
        saved = loaded_from = None
        if args.resume:
            saved, loaded_from = _load_resume(run_dir, require_complete=profile)
            if saved is None:
                if profile:
                    raise FileNotFoundError(f"Requested {experiment.PROFILE} resume has no recoverable checkpoint: {run_dir}")
                print(f"[{name}] no resume checkpoint; start this arm from step 0", flush=True)
                args.resume = False
        _set_seed(args.seed, False)
        if args.device == "cuda" and not th.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        runner = ParallelRunner(args, logger)
        env_info = runner.get_env_info()
        for key, value in env_info.items():
            setattr(args, key, value)
        if saved is not None:
            saved_config = saved["config"]
            current = vars(args)
            mutable = RESUME_MUTABLE - {"implementation_revision"} if profile else RESUME_MUTABLE
            differing = sorted(key for key in set(saved_config) | set(current)
                               if key not in mutable
                               and saved_config.get(key, RESUME_DEFAULTS.get(key)) != current.get(key))
            if differing:
                raise ValueError(
                    "Resume configuration differs: " + ", ".join(differing)
                    + ". Old replay data may not be mixed with a changed experiment;"
                    " start a new --run instead.")
            if profile and loaded_from.name == "resume.pt.pending":
                # A readable pending snapshot can be the only surviving full
                # save. Commit it before the next save reuses the pending path.
                committed = run_dir / "resume.pt"
                os.replace(loaded_from, committed)
                _fsync_dir(run_dir)
                loaded_from = committed
            safe_checkpoint_path = loaded_from
            safe_checkpoint_t_env = saved_t_env = int(saved["progress"]["t_env"])
        scheme, groups, preprocess = make_scheme(env_info, multi_task=args.multi_task)
        # ALMA controllers expect the base feature dimension as a scalar.
        model_scheme = copy.deepcopy(scheme)
        model_scheme["entities"]["vshape"] = args.entity_shape
        model_scheme["actions_onehot"] = {"vshape": (args.n_actions,), "group": "agents"}
        mac = build_mac(model_scheme, groups, args)
        runner.setup(scheme, groups, preprocess, mac)
        learner = build_learner(mac, model_scheme, logger, args)
        learner_ref["learner"] = learner
        if args.device == "cuda":
            th.cuda.manual_seed_all(args.seed)
            learner.cuda()
        replay = ReplayBuffer(scheme, groups, args.buffer_size, args.episode_limit + 1,
                              preprocess=preprocess, device="cpu", efficient_store=True, max_traj_len=-1)
        if args.resume:
            if saved is not None:
                if loaded_from.name != "resume.pt":
                    print(f"[{name}] {loaded_from.name} used; resume.pt was missing or unreadable", flush=True)
                _load_network_state(learner, saved["networks"])
                replay = saved["replay"]
                replay.to("cpu")
                runner.load_state_dict(saved["runner"])
                state.update(saved["progress"])
                if "validation_completed_episodes" not in saved["progress"]:
                    state["validation_completed_episodes"] = (int(state["next_eval"]) - 1) * (100 if args.env == "had" else 128)
                if not {"validation_measured_episodes", "validation_measured_seconds"} <= saved["progress"].keys():
                    # Older snapshots know coverage, but not its matching time.
                    # Never combine reused CSV episodes with new timing data.
                    state.update(validation_measured_episodes=0, validation_measured_seconds=0.0)
                random.setstate(saved["rng"]["python"])
                np.random.set_state(saved["rng"]["numpy"])
                th.set_rng_state(saved["rng"]["torch"].cpu())
                if args.device == "cuda":
                    th.cuda.set_rng_state_all([item.cpu() for item in saved["rng"]["cuda"]])
                saved_t_env = runner.t_env
                print(f"[{name}] resume t_env={runner.t_env:,} from {loaded_from.name}", flush=True)
        del saved
        session_start_t_env = runner.t_env
        state["status"] = "running"
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
        sample_resources()

        def save(kind, full=False):
            nonlocal saved_t_env, last_save, safe_checkpoint_path, safe_checkpoint_t_env
            if profile and full and not transaction_complete:
                raise RuntimeError("Cannot save a partial collection/update transaction as a valid resume")
            value = dict(config=vars(args), env_info=env_info, networks=_network_state(learner),
                         progress={**state, "t_env": runner.t_env,
                                   "elapsed_seconds": state["elapsed_seconds"] + time.monotonic() - started})
            if profile:
                if kind == "final":
                    if "final_artifact_id" not in state:
                        state["final_artifact_id"] = experiment.new_artifact_id()
                    value["artifact_id"] = state["final_artifact_id"]
                    value["progress"]["final_artifact_id"] = state["final_artifact_id"]
                else:
                    value["artifact_id"] = experiment.new_artifact_id()
            if full:
                value.update(transaction_complete=True, replay=replay, runner=runner.state_dict(),
                             rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                      torch=th.get_rng_state(),
                                      cuda=th.cuda.get_rng_state_all() if args.device == "cuda" else []))
            path = run_dir / f"{kind}.pt"
            _atomic_save(value, path, keep_previous=(kind == "resume" and full))
            if full:
                saved_t_env = runner.t_env
                safe_checkpoint_path, safe_checkpoint_t_env = path, int(runner.t_env)
                last_save = time.monotonic()
                if kind == "resume":
                    print(f"[{name}] saved resume t_env={runner.t_env:,}", flush=True)
            return path

        def heartbeat(evaluating=False, done=0, total=0, in_flight=0):
            nonlocal last_progress, stop_requested, collection_steps_seen
            if transaction_phase == "collecting" and not evaluating:
                collection_steps_seen = max(collection_steps_seen, int(in_flight))
            if ((stop_event is not None and stop_event.is_set()) or (output / "stop.request").exists()
                    or (profile and (run_dir / "pause.request").exists())):
                stop_requested = True
            now = time.monotonic()
            if now - last_progress < args.progress_interval:
                return
            elapsed = state["elapsed_seconds"] + now - started
            session = now - started
            session_sps = (runner.t_env - session_start_t_env) / max(session, 1e-6)
            metrics = dict(status="evaluating" if evaluating else "training", phase=args.env,
                           t_env=runner.t_env, budget_steps=args.t_max,
                           steps_per_second=session_sps,
                           lifetime_steps_per_second=runner.t_env / max(elapsed, 1e-6),
                           session_elapsed_seconds=session,
                           eval_completed=eval_progress["completed"] + done if evaluating else 0,
                           eval_total=eval_progress["total"] if evaluating else 0,
                           latest_validation_D=state.get("latest_validation_D"),
                           latest_validation_win_rate=state.get("latest_validation_win_rate"),
                           checkpoint_t_env=saved_t_env,
                           updates=state["updates"], elapsed_seconds=elapsed,
                           loss=getattr(learner, "last_metrics", {}).get("loss"))
            metrics.update(sample_resources())
            record.progress(**metrics)
            loss_label = "-" if metrics["loss"] is None else f"{metrics['loss']:.4f}"
            validation_key = "latest_validation_D" if args.env == "had" else "latest_validation_win_rate"
            d_label = "-" if state.get(validation_key) is None else f"{state[validation_key]:.3f}"
            print(f"[{name}] {int(session)//3600}:{int(session)%3600//60:02d}:{int(session)%60:02d}  "
                  f"{runner.t_env:,}/{args.t_max:,}  {metrics['steps_per_second']:.1f}/s  "
                  f"upd={state['updates']}  L={loss_label}"
                  + (f"  ev={metrics['eval_completed']}/{metrics['eval_total']}" if evaluating else "")
                  + f"  {'D' if args.env == 'had' else 'win'}={d_label}", flush=True)
            last_progress = now

        runner.progress_callback = heartbeat

        def values(batch, t, actions, active):
            # Use the outputs already computed by select_actions: do not advance a GRU twice.
            if hasattr(mac, "evaluation_values"):
                return mac.evaluation_values(batch, t, actions, active, learner.mixer)
            return {"q_tot": [None] * len(active), "q_i": [None] * len(active)}

        runner.value_callback = values

        def evaluate_jobs(jobs):
            nonlocal validation_clock
            # Validation must not consume the future training random streams.
            rng = (random.getstate(), np.random.get_state(), th.get_rng_state(),
                   th.cuda.get_rng_state_all() if args.device == "cuda" else [])
            runner_state = runner.state_dict()
            previous = read_records(output, "episodes", run=args.run, method=name, seed=args.seed,
                                    env=args.env if profile else None)
            pending = remaining_jobs(jobs, previous)
            eval_progress.update(completed=len(jobs) - len(pending), total=len(jobs))
            validation_clock = time.monotonic()
            validation_base = (int(jobs[0].get("eval_point", 1)) - 1) * len(jobs) if jobs else 0
            if profile:
                state["validation_completed_episodes"] = validation_base + len(jobs) - len(pending)
            fresh = []
            try:
                for offset in range(0, len(pending), args.batch_size_run):
                    eval_progress["completed"] = len(jobs) - len(pending) + offset
                    group = pending[offset:offset + args.batch_size_run]
                    measurement_started = time.monotonic()
                    _, summaries = runner.run(test_mode=True, jobs=group)
                    rows = [{**summary, **job, "train_dist": "mixed_le10"}
                            for summary, job in zip(summaries, group)]
                    for job, trajectory in zip(group, runner.last_trajectories):
                        if job.get("retain_trajectory") and trajectory is not None:
                            record.trajectories([{**job, "trajectory": trajectory}])
                    record.episodes(rows)
                    if profile:
                        eval_progress["completed"] += len(rows)
                        state["validation_completed_episodes"] = validation_base + eval_progress["completed"]
                        # Commit the pair only after the corresponding rows are
                        # stored. Reused rows and interrupted groups add neither.
                        state["validation_measured_episodes"] += len(rows)
                        state["validation_measured_seconds"] += time.monotonic() - measurement_started
                    fresh.extend(rows)
                    if stop_requested:
                        break
                return previous + fresh
            finally:
                state["validation_seconds"] += time.monotonic() - validation_clock
                validation_clock = None
                random.setstate(rng[0]); np.random.set_state(rng[1]); th.set_rng_state(rng[2])
                if args.device == "cuda":
                    th.cuda.set_rng_state_all(rng[3])
                runner.load_state_dict(runner_state)

        thresholds = ((experiment.validation_thresholds(args.env) if profile else
                       evaluation_thresholds(args.t_max)) if formal else [])
        def run_pending_evaluations():
            nonlocal transaction_phase
            while (formal and not stop_requested and state["next_eval"] <= len(thresholds)
                   and runner.t_env >= thresholds[state["next_eval"] - 1]):
                # Save this exact evaluation policy before starting a resumable validation point.
                save("resume", full=True)
                transaction_phase = "validating"
                point = state["next_eval"]
                jobs = (experiment.validation_jobs(args.env, point, runner.t_env) if profile
                        else validation_jobs(point, runner.t_env))
                rows = evaluate_jobs(jobs)
                current = [row for row in rows if row.get("phase") == "train_eval" and row.get("eval_point") == point]
                score = experiment.validation_score(args.env, current) if profile else validation_score(current)
                if score is None:
                    if stop_requested:
                        break
                    raise RuntimeError(f"Validation point {point} is incomplete")
                state["latest_validation_D" if args.env == "had" else "latest_validation_win_rate"] = score
                better = score > state["best_score"] if profile and args.env == "smacv2" else score < state["best_score"]
                if better:
                    # Commit the weights first: a failed write must not leave a
                    # recorded score that no retained model can reproduce.
                    save("best")
                    state["best_score"] = score
                    state["best_t_env"] = runner.t_env
                state["next_eval"] += 1
                transaction_phase = "idle"
                refresh_report_safe()
                if stop_requested:
                    break

        if profile and safe_checkpoint_path is None:
            # Even an OOM in the very first environment/actor interaction can
            # return to the original weights, empty replay and exact seed state.
            save("resume", full=True)
            sample_resources()
        transaction_phase = "idle"
        run_pending_evaluations()
        while runner.t_env < args.t_max and not stop_requested:
            cycle_start = time.monotonic()
            updates_before_batch = int(state["updates"])
            transaction_complete = False
            transaction_phase = "collecting"
            collection_steps_seen = 0
            episode_batch, summaries = runner.run(max_train_steps=args.t_max - runner.t_env)
            transaction_phase = "updating"
            collection_steps_seen = 0
            collected_at = time.monotonic()
            episode_batch.to("cpu")
            replay.insert_episode_batch(episode_batch)
            state["episode_num"] += episode_batch.batch_size
            update_metrics = []
            if replay.can_sample(args.batch_size):
                updates_due = episode_batch.batch_size if name == "transfqmix" else args.training_iters
                for _ in range(updates_due):
                    heartbeat()
                    if stop_requested and not profile and name != "transfqmix":
                        break
                    sample = replay.sample(args.batch_size)
                    max_t = int(sample.max_t_filled().item())
                    sample = sample[:, :max_t]
                    sample.to(args.device)
                    metrics = learner.train(sample, runner.t_env, state["episode_num"])
                    state["updates"] += 1
                    first_session_update = not observed_update
                    observed_update = True
                    if profile and first_session_update:
                        sample_resources()
                    if isinstance(metrics, dict):
                        update_metrics.append(metrics)
                    heartbeat()
                    if stop_requested and not profile and name != "transfqmix":
                        break
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
            transaction_complete = True
            transaction_phase = "idle"
            session_complete_batches += 1
            if profile and (state["updates"] == 0 or
                            (state["updates"] > 0 and
                             (updates_before_batch == 0 or session_complete_batches == 1))):
                # Warmup replay must survive a first-update OOM, and the first
                # fully updated batch establishes a recovery/admission boundary.
                save("resume", full=True)
                sample_resources()
            metrics = dict(logger.latest)
            if update_metrics:
                for key in update_metrics[-1]:
                    metrics[key] = float(np.mean([item[key] for item in update_metrics if key in item]))
                norms = [float(item["grad_norm"]) for item in update_metrics if "grad_norm" in item]
                if norms:
                    # Spike: >10, or >20x the running median of the previous 200 updates.
                    history = list(state.get("grad_norm_history", []))
                    for value in norms:
                        median = float(np.median(history)) if history else None
                        if value > 10 or (median is not None and value > 20 * median):
                            state["grad_spikes"] = int(state.get("grad_spikes", 0)) + 1
                        history = (history + [value])[-200:]
                    state["grad_norm_history"] = history
                    metrics["grad_norm_max"] = max(norms)
                    metrics["grad_spikes"] = int(state.get("grad_spikes", 0))
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
            elif args.env == "smacv2":
                metrics["train_battle_won"] = float(np.mean([s["battle_won"] for s in summaries]))
            record.learning(runner.t_env, metrics)
            run_pending_evaluations()
            heartbeat()
            if time.monotonic() - last_save >= args.resume_interval:
                save("resume", full=True)
                if profile:
                    sample_resources()
            if time.monotonic() - last_report >= args.report_interval:
                refresh_report_safe()
                last_report = time.monotonic()

        state["status"] = "stopped" if stop_requested else "completed"
        if not stop_requested:
            save("final")
            if formal and not profile and not bool(getattr(args, "skip_final_eval", False)):
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
        refresh_report_safe()
        return str(path if stop_requested else run_dir / "final.pt")
    except BaseException as error:
        oom = _cuda_oom_details(error)
        state["status"] = "failed"
        if not profile and runner is not None and "save" in locals():
            try:
                save("resume", full=True)
            except BaseException:
                pass
        if profile:
            # Never run save() here on OOM: replay, optimizer, targets, or the
            # worker RNG may already belong to a partially advanced transaction.
            # CPU metadata and allocator counters do not need fresh GPU tensors.
            try:
                sample_resources()
            except Exception:
                pass
            observed_t_env = max(int(runner.t_env) if runner is not None else 0,
                                 int(safe_checkpoint_t_env)) + collection_steps_seen
            automatic = bool(oom is not None and safe_checkpoint_path is not None and
                             safe_checkpoint_path.is_file())
            failure = dict(kind="cuda_oom" if oom is not None else "exception",
                           automatic_recovery=automatic,
                           recovery_status="safe_checkpoint" if automatic else "start_failure" if oom else "failed",
                           env=args.env, method=name, seed=int(args.seed), pid=os.getpid(),
                           physical_gpu=_physical_gpu(), updated_at=time.time(),
                           transaction_phase=transaction_phase, transaction_complete=transaction_complete,
                           observed_t_env=observed_t_env, t_env=observed_t_env,
                           observed_t_env_is_lower_bound=transaction_phase == "collecting",
                           safe_checkpoint_t_env=int(safe_checkpoint_t_env),
                           safe_checkpoint_path=None if safe_checkpoint_path is None else str(safe_checkpoint_path.resolve()),
                           error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                           cuda_peak_reserved_gib=resource_latest.get("cuda_peak_reserved_gib", 0.),
                           cuda_reserved_gib=resource_latest.get("cuda_reserved_gib", 0.),
                           requested_allocation_gib=None, minimum_additional_gib=None)
            if oom is not None:
                failure.update(oom)
            experiment.atomic_json(run_dir / "failure.json", failure)
        record.progress(status="failed", error=f"{type(error).__name__}: {error}",
                        t_env=runner.t_env if runner else 0, checkpoint_t_env=saved_t_env)
        refresh_report_safe()
        raise
    finally:
        if runner is not None:
            runner.close_env()
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)


def load_policy(name, checkpoint):
    setup_runtime()
    from open_score.models import build_mac
    import torch as th
    saved = th.load(checkpoint, map_location="cpu", weights_only=False)
    saved_method = canonical_method(saved["config"]["method"])
    requested = canonical_method(name)
    if saved_method != requested and not (requested == "alma_legacy" and saved_method == "alma"):
        raise ValueError("Checkpoint method does not match requested policy")
    args = SimpleNamespace(**saved["config"])
    args.device = "cpu"
    args.use_cuda = False
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


__all__ = ["train", "load_policy", "load_config", "canonical_method", "METHODS", "PROBE_METHODS",
           "V4_METHODS", "V5_METHODS", "V5_GLOBAL_METHODS", "POLICY_METHODS",
           "MAIN_METHODS", "MAIN_ABLATION_METHODS", "MAIN_TRAIN_METHODS", "MAIN_OVERRIDES"]
