"""Native terminal simulation with private branch streams and exact snapshots."""
from __future__ import annotations

import copy
import random
import time
import numpy as np

from open_score.research_v4.environment import make_env as original_make_env
from open_score.research_v4.actions import rule_grouping
from .protocol import EpisodeSpec


def make_env(spec):
    if isinstance(spec, dict):
        spec = EpisodeSpec.from_dict(spec)
    env = original_make_env(spec.red_count, spec.blue_count, opponent=spec.opponent,
                            seed=spec.opening_seed, max_steps=50, command_interval=5)
    env.reset(seed=spec.opening_seed)
    env.set_rng(spec.opponent_seed)
    env.episode_spec = spec
    return env


def env_from_snapshot(snapshot):
    physical = snapshot['physical']
    env = original_make_env(len(physical.red_assignment), len(physical.blue_assignment),
                           opponent=snapshot['opponent'], seed=0,
                           max_steps=snapshot['max_steps'], command_interval=snapshot['command_interval'],
                           targets=len(physical.active_target_ids))
    env.reset(seed=0)
    env.restore(snapshot)
    return env


def choose(policy, env):
    if policy is None:
        return rule_grouping(env.state())
    if hasattr(policy, 'act_env'):
        value = policy.act_env(env)
    elif hasattr(policy, 'act'):
        value = policy.act(env.state())
    else:
        value = policy(env.state())
    return value.action if hasattr(value, 'action') else value


def paired_rollouts(snapshot, candidates, continuation=None, branch_seeds=None,
                    continuation_version='rule_grouping_v1'):
    seeds = list(map(int, [] if branch_seeds is None else branch_seeds))
    if not seeds or len(set(seeds)) != len(seeds) or not candidates:
        raise ValueError('Need candidates and unique branch seeds')
    if continuation is not None and continuation_version == 'rule_grouping_v1':
        raise ValueError('A custom continuation requires its frozen version')
    env = env_from_snapshot(snapshot)
    ambient = (random.getstate(), np.random.get_state())
    state = env.state()
    rows = []
    try:
        for ci, candidate in enumerate(candidates):
            candidate.validate(state.ids('red'), state.ids('targets'), max_members=None)
            outcomes, physical_steps, times, records = [], [], [], []
            for seed in seeds:
                env.restore(snapshot)
                env.set_rng(seed)
                random.seed(seed)
                np.random.seed(seed)
                if hasattr(continuation, 'reset'):
                    continuation.reset()
                started = time.perf_counter()
                following, _, done, info = env.step(candidate)
                first_step, first_event = following.step, info['event_reason']
                work = info['delta']
                while not done:
                    following, _, done, info = env.step(choose(continuation, env))
                    work += info['delta']
                success = int(info['success'])
                outcomes.append(success)
                physical_steps.append(int(work))
                times.append(int(following.step-state.step))
                records.append(dict(branch_seed=seed, success_native=success,
                    physical_steps_simulated=int(work), terminal_step=following.step,
                    first_action_end_step=first_step, first_action_end_event_type=first_event,
                    termination_reason=info['event_reason'], wall_time_s=time.perf_counter()-started,
                    completed=True, continuation_policy_id=continuation_version))
            rows.append(dict(candidate_id=ci, action=candidate.to_dict(), state=state.to_dict(),
                y=float(np.mean(outcomes)), outcomes=outcomes, physical_steps=physical_steps,
                terminal_steps=times, branch_seeds=seeds, branches=records,
                continuation_version=continuation_version,
                label_semantics='first_plan_one_event_then_frozen_continuation_to_native_terminal'))
    finally:
        env.close()
        random.setstate(ambient[0])
        np.random.set_state(ambient[1])
    return rows


def step_with_trace(env, action):
    """Instrument real execution only; preserve the original physical transition."""
    old_step = env.adapter.step
    frames = []
    def record(actions, *args, **kwargs):
        alive = {i for i, r in env.adapter.agent_states('Red').items() if r['alive']}
        reserve = len(alive.intersection(env.previous.reserve))
        result = old_step(actions, *args, **kwargs)
        frames.append(dict(physical_step=env.adapter.step_count, red_alive_before=len(alive),
                           reserve_alive_before=reserve,
                           target_hp={str(i): float(r['health']) for i, r in env.adapter.target_states().items()},
                           actions={str(k): int(v) for k, v in actions.items()}))
        return result
    env.adapter.step = record
    try:
        state, reward, done, info = env.step(action)
        info = {**info, 'physical_trace': frames}
        return state, reward, done, info
    finally:
        env.adapter.step = old_step
