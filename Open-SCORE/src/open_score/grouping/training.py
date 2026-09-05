"""Event-level PPO with actual physical-step budgets and resumable checkpoints."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import signal
import time

import numpy as np
import torch

from .environment import KnownOpponentEnv
from .policy import GroupingPolicy
from .storage import (atomic_checkpoint, atomic_json, append_jsonl, fingerprint,
                      provenance, random_state, restore_random_state, seed_everything)


METHODS = ('static', 'dlom', 'full', 'random', 'selective', 'alma', 'rule')


def make_policy(config):
    method = config.get('method', 'selective')
    model = dict(config.get('model', {}))
    if method == 'alma':
        from .baselines import AlmaStylePolicy
        return AlmaStylePolicy(**model, seed=config.get('seed', 20260905)).to(config.get('device', 'cpu'))
    if method in ('static', 'dlom'):
        from .baselines import StaticPolicy, DLOMSearchPolicy
        return StaticPolicy() if method == 'static' else DLOMSearchPolicy(device=config.get('device', 'cpu'))
    return GroupingPolicy(mode=method, **model).to(config.get('device', 'cpu'))


def contract(config):
    keys = ('method', 'seed', 'opponent', 'train_scales', 'max_steps', 'command_interval',
            'model', 'ppo', 'release_distribution')
    return {key: config.get(key) for key in keys}


def load_policy(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = dict(payload['config'])
    config['device'] = device
    policy = make_policy(config)
    policy.load_state_dict(payload['model'])
    if config['method'] == 'random' and payload.get('release_distribution'):
        policy.set_release_distribution(payload['release_distribution'])
    policy.eval()
    return policy, payload


def smdp_gae(rewards, values, next_values, deltas, dones, gamma=1.0, lam=0.95):
    """Lambda is applied per upper event; actual task horizons are dones."""
    advantages = np.zeros(len(rewards), dtype=np.float64)
    following = 0.0
    for index in reversed(range(len(rewards))):
        discount = gamma ** int(deltas[index])
        live = 1.0 - float(dones[index])
        residual = rewards[index] + live * discount * next_values[index] - values[index]
        following = residual + live * discount * lam * following
        advantages[index] = following
    return advantages, advantages + np.asarray(values)


def ppo_update(policy, optimizer, rollout, settings):
    device = next(policy.parameters()).device
    advantages, returns = smdp_gae(
        *[[row[key] for row in rollout]
          for key in ('reward', 'value', 'next_value', 'delta', 'done')],
        gamma=float(settings.get('gamma', 1.0)), lam=float(settings.get('gae_lambda', .95)))
    advantage = torch.tensor(advantages, dtype=torch.float32, device=device)
    if len(rollout) > 1 and float(advantage.std(unbiased=False)) > 1e-8:
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    target = torch.tensor(returns, dtype=torch.float32, device=device)
    old_log = torch.tensor([r['log_prob'] for r in rollout], dtype=torch.float32, device=device)
    records, by_size = [], {}
    batch_size = int(settings.get('batch_size', 16))
    clip = float(settings.get('clip', .2))
    for _ in range(int(settings.get('epochs', 2))):
        order = np.random.permutation(len(rollout))
        stop = False
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size].tolist()
            evaluated = [policy.evaluate_action(rollout[i]['state'], rollout[i]['trace']) for i in indices]
            logp, entropy, value = (torch.stack([row[k].reshape(()) for row in evaluated]) for k in range(3))
            difference = logp - old_log[indices]
            if not torch.isfinite(difference).all():
                raise FloatingPointError('Non-finite joint policy log probability')
            ratio = torch.exp(difference.clamp(-60, 60))
            unclipped = ratio * advantage[indices]
            clipped = ratio.clamp(1 - clip, 1 + clip) * advantage[indices]
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = .5 * (value - target[indices]).square().mean()
            loss = actor_loss + float(settings.get('value_coef', .5)) * value_loss
            loss = loss - float(settings.get('entropy_coef', 0.0)) * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            kl = float(((ratio - 1) - difference).mean().detach())
            clipped_flags = ((ratio - 1).abs() > clip).detach().cpu().numpy()
            for j, i in enumerate(indices):
                size = len(rollout[i]['released_ids'])
                by_size.setdefault(str(size), []).append(float(clipped_flags[j]))
            records.append({'loss': float(loss.detach()), 'policy_loss': float(actor_loss.detach()),
                            'value_loss': float(value_loss.detach()), 'entropy': float(entropy.mean().detach()),
                            'approx_kl': kl, 'clip_fraction': float(clipped_flags.mean()),
                            'gradient_norm': float(gradient_norm)})
            if kl > float(settings.get('target_kl', .05)):
                stop = True
                break
        if stop:
            break
    metrics = {key: float(np.mean([row[key] for row in records])) for key in records[0]}
    metrics['clip_fraction_by_release_count'] = {key: float(np.mean(value)) for key, value in by_size.items()}
    return metrics


def train(config, output, *, steps=2000, wall_seconds=300.0, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if config['method'] in ('static', 'dlom'):
        raise ValueError('Static and DLOM policies do not train')
    if steps is not None and steps < 1:
        raise ValueError('steps must be positive or None')
    if wall_seconds is not None and wall_seconds <= 0:
        raise ValueError('wall_seconds must be positive or None')
    seed_everything(config['seed'])
    torch.set_num_threads(int(config.get('torch_threads', 1)))
    evidence = provenance()
    config_hash = fingerprint(contract(config))
    latest = output / 'latest.pt'
    if latest.exists() and not resume:
        raise FileExistsError(f'{latest} exists; use --resume or a different output directory')
    policy = make_policy(config)
    if config.get('release_distribution') and config['method'] == 'random':
        policy.set_release_distribution(config['release_distribution'])
    optimizer = None if config['method'] == 'alma' else torch.optim.Adam(
        policy.parameters(), lr=float(config.get('ppo', {}).get('learning_rate', 3e-4)))
    counters = {'physical_steps': 0, 'upper_events': 0, 'decode_steps': 0,
                'episodes': 0, 'updates': 0, 'next_episode': 0, 'training_seconds': 0.0}
    release_counts = Counter()
    initial_model = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    if latest.exists():
        payload = torch.load(latest, map_location=config.get('device', 'cpu'), weights_only=False)
        if payload['config_hash'] != config_hash:
            raise ValueError('Resume configuration differs from the trained task/method')
        if payload['provenance']['source_hash'] != evidence['source_hash']:
            raise ValueError('Resume source code differs; preserve this run and choose a new output directory')
        policy.load_state_dict(payload['model'])
        if optimizer is not None:
            optimizer.load_state_dict(payload['optimizer'])
        else:
            policy.load_training_state_dict(payload['training_state'])
        counters.update(payload['counters'])
        release_counts.update({int(k): v for k, v in payload.get('release_counts', {}).items()})
        initial_model = payload['initial_model']
        restore_random_state(payload['random_state'])
    start_steps, started = counters['physical_steps'], time.perf_counter()
    previous_seconds = counters['training_seconds']
    first_parameters = [p.detach().clone() for p in policy.parameters()]
    settings = config.get('ppo', {})
    rollout_size = int(settings.get('rollout_events', 64))
    rollout, episode_returns, environments = [], [], {}
    last_snapshot = started
    state, env, episode_start, episode_reward = None, None, None, 0.0
    last_metrics = {}
    stop_requested = False
    old_signal_handler = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print('Stop requested; finishing the current safe update boundary.', flush=True)

    signal.signal(signal.SIGINT, request_stop)

    def save(name='latest.pt'):
        counters['training_seconds'] = previous_seconds + time.perf_counter() - started
        payload = {'schema': 'known-grouping-v2', 'config': config, 'config_hash': config_hash,
                   'model': policy.state_dict(), 'initial_model': initial_model,
                   'optimizer': optimizer.state_dict() if optimizer is not None else None,
                   'training_state': policy.training_state_dict() if optimizer is None else None,
                   'random_state': random_state(), 'counters': dict(counters),
                   'release_counts': dict(release_counts),
                   'release_distribution': config.get('release_distribution'),
                   'provenance': evidence,
                   'resume_semantics': 'Complete updates retained; resume starts a fresh episode and discards partial rollout.'}
        atomic_checkpoint(output / name, payload)

    if not (output / 'initialized.pt').exists():
        save('initialized.pt')
    atomic_json(output / 'manifest.json', {'config': config, 'config_hash': config_hash, 'provenance': evidence})

    def budget_reached():
        return (stop_requested or (steps is not None and counters['physical_steps'] - start_steps >= steps)
                or (wall_seconds is not None and time.perf_counter() - started >= wall_seconds))

    def update_rollout():
        nonlocal rollout, last_metrics
        if not rollout:
            return
        if optimizer is not None:
            last_metrics = ppo_update(policy, optimizer, rollout, settings)
            counters['updates'] += 1
        rollout = []
        elapsed = time.perf_counter() - started
        row = dict(counters, **last_metrics, elapsed_seconds=elapsed,
                   steps_per_second=(counters['physical_steps'] - start_steps) / max(elapsed, 1e-9),
                   recent_success_rate=float(np.mean(episode_returns[-20:])) if episode_returns else None)
        append_jsonl(output / 'training.jsonl', row)
        atomic_json(output / 'status.json', dict(row, status='training'))
        save()
        print(f"[{config['method']}] steps={counters['physical_steps']} updates={counters['updates']} "
              f"episodes={counters['episodes']} speed={row['steps_per_second']:.1f}/s "
              f"success20={row['recent_success_rate']}", flush=True)

    interrupted = False
    try:
        while not budget_reached():
            if state is None:
                scale = int(config['train_scales'][counters['next_episode'] % len(config['train_scales'])])
                if scale not in environments:
                    environments[scale] = KnownOpponentEnv(
                        red=scale, blue=scale, max_steps=config['max_steps'], opponent=config['opponent'],
                        device=config.get('executor_device', 'cpu'), command_interval=config['command_interval'])
                env = environments[scale]
                episode_seed = int(config['seed']) + 100000 + counters['next_episode']
                counters['next_episode'] += 1
                state = env.reset(seed=episode_seed)
                episode_start, episode_reward = counters['physical_steps'], 0.0
            with torch.no_grad():
                decision = policy.act(state)
            decision.action.validate(state.ids('red'), state.ids('targets'))
            next_state, reward, done, info = env.step(decision.action)
            delta = int(info['delta'])
            if delta < 1:
                raise AssertionError('Macro action did not advance physical time')
            counters['physical_steps'] += delta
            counters['upper_events'] += 1
            counters['decode_steps'] += decision.trace.get('decode_steps',
                len(decision.trace.get('selection', [])) + len(decision.trace.get('repairs', [])))
            if state.step != 0:
                release_counts[len(decision.released_ids)] += 1
            with torch.no_grad():
                next_value = 0.0 if done or optimizer is None else float(policy.value(next_state))
            rollout.append({'state': state, 'trace': decision.trace, 'log_prob': float(decision.log_prob),
                            'value': float(decision.value), 'next_value': next_value, 'reward': float(reward),
                            'delta': delta, 'done': bool(done), 'released_ids': decision.released_ids})
            if optimizer is None:
                policy.observe(state, decision.action, reward, delta, done, next_state)
                update_metrics = policy.update()
                if update_metrics:
                    last_metrics = update_metrics
                    counters['updates'] += 1
            episode_reward += float(reward)
            state = next_state
            if done:
                counters['episodes'] += 1
                episode_returns.append(episode_reward)
                append_jsonl(output / 'episodes.jsonl', {'episode': counters['episodes'],
                    'physical_steps': counters['physical_steps'], 'length': counters['physical_steps'] - episode_start,
                    'success': bool(info['success']), 'return': episode_reward, 'scale': scale})
                state = None
            if len(rollout) >= rollout_size:
                update_rollout()
            if time.perf_counter() - last_snapshot >= 60 and counters['updates']:
                save(f"snapshots/step_{counters['physical_steps']}.pt")
                last_snapshot = time.perf_counter()
        update_rollout()
    except KeyboardInterrupt:
        interrupted = True
        # The last completed update is the safe resume boundary.
        rollout.clear()
        print('Interrupted; saving completed model updates.', flush=True)
    finally:
        save()
        signal.signal(signal.SIGINT, old_signal_handler)
        for instance in environments.values():
            instance.close()
    elapsed = time.perf_counter() - started
    delta_parameters = math.sqrt(sum(float((p.detach() - old).square().sum())
                                     for p, old in zip(policy.parameters(), first_parameters)))
    initial_delta = math.sqrt(sum(float((value.detach().cpu() - initial_model[key]).square().sum())
                                  for key, value in policy.state_dict().items() if value.is_floating_point()))
    result = dict(counters, elapsed_seconds=elapsed, invocation_steps=counters['physical_steps'] - start_steps,
                  steps_per_second=(counters['physical_steps'] - start_steps) / max(elapsed, 1e-9),
                  parameter_change_l2=delta_parameters, change_from_initial_l2=initial_delta,
                  requested_steps=steps, requested_seconds=wall_seconds,
                  time_overrun_seconds=max(0., elapsed - wall_seconds) if wall_seconds else 0.,
                  status='interrupted' if interrupted or stop_requested else 'complete', latest_checkpoint=str(latest.resolve()),
                  release_counts=dict(release_counts), metrics=last_metrics)
    if provenance()['assets'] != evidence['assets']:
        raise AssertionError('Frozen assets changed during training')
    atomic_json(output / 'status.json', result)
    return result
