# T3 reproduction and migration map

Independent actor/critic PPO; matched candidate and unrestricted autoregressive grouping; PPO clipping uses joint event likelihood.

Scope: known reactive opponent; shared rule executor; identity-level grouping and reserve; 15 mixed cells; native 50-step terminal success.
This is a mechanism reimplementation/transfer. Original benchmark environments are not run and original-paper scores are not claimed.
Parameters below are frozen before outcomes. Differences from source methods are necessary to represent variable groups, physical command events and known-opponent continuation.
Correctness evidence: tests/test_v5_planning.py, test_v5_neural.py, test_v5_learning_tasks.py, test_v5_value_tasks.py and test_v5_runtime.py.

```json
{
  "version": "known-rule-mixed-v5.1",
  "seed": 20260907,
  "smoke": true,
  "multiplier": 1.0,
  "cells": [
    [
      8,
      4
    ],
    [
      12,
      9
    ],
    [
      16,
      16
    ]
  ],
  "opponent": "reactive",
  "executor": "rule_group_v1",
  "max_steps": 50,
  "command_interval": 5,
  "gamma": 1.0,
  "device": "auto",
  "threads": 1,
  "cpu_quotas": {
    "T1": 1,
    "T2": 1,
    "T3": 1,
    "T4": 1,
    "T5": 1,
    "T6": 1
  },
  "model": {
    "hidden_dim": 32,
    "layers": 1,
    "heads": 4
  },
  "eval_per_cell": 1,
  "validation_per_cell": 1,
  "bc_episodes": 3,
  "bc_epochs": 1,
  "diagnostic_states": 3,
  "own_diagnostic_states": 1,
  "selection_branches": 2,
  "verification_branches": 2,
  "latency_states": 3,
  "teacher_validation_episodes": 3,
  "t1": {
    "candidates": 4,
    "branches": 2,
    "control_episodes": 3
  },
  "t2": {
    "iterations": 4,
    "depth": 2,
    "k_action": 1.5,
    "alpha_action": 0.5,
    "k_state": 1.0,
    "alpha_state": 0.5,
    "exploration": 1.0
  },
  "t3": {
    "steps": 120,
    "rollout": 32,
    "batch_size": 16,
    "epochs": 1,
    "learning_rate": 0.0003,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "max_gradient_norm": 0.5,
    "target_kl": 0.03,
    "entropy_start": 0.02,
    "entropy_end": 0.005,
    "candidates": 8,
    "validation_fractions": [
      0.0,
      1.0
    ]
  },
  "t4": {
    "rounds": 2,
    "states_per_round": 4,
    "candidates": 4,
    "branches": 2,
    "epochs": 1,
    "batch_size": 2,
    "learning_rate": 0.0003
  },
  "t5": {
    "states": 4,
    "episodes": 12,
    "branches": 2,
    "batch_size": 4,
    "buffer_size": 10000,
    "target_interval": 4,
    "learning_rate": 0.0003,
    "max_gradient_norm": 40.0,
    "epsilon_steps": 10000,
    "epsilon_end": 0.01
  },
  "t6": {
    "train_states": 4,
    "validation_states": 3,
    "candidates": 4,
    "branches": 2,
    "epochs": 2,
    "batch_size": 2,
    "learning_rate": 0.0003,
    "search_budget": 8,
    "thresholds": [
      0.0,
      0.005,
      0.01,
      0.02
    ]
  }
}
```

<!-- PINNED_REFERENCES -->

## Pinned reference provenance

Mechanisms are reimplemented/adapted in this environment. Original paper environments and full original algorithms are not claimed as reproduced.

- [ppo](https://arxiv.org/abs/1707.06347): paper reference; no source code copied; event_joint_action_PPO_reimplementation.
- [attention](https://github.com/wouterkool/attention-learn-to-route): c9abf41ac2f878a55b20dc7e829bc942bb999631; MIT; construction representation adapted; original REINFORCE optimizer not replicated.
- [masking](https://arxiv.org/abs/2006.14171): paper reference; no source code copied; legal conditional action masks.
