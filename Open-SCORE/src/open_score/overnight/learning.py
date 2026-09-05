"""Batched PPO and replay-based Double DQN for complete candidate actions."""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F


def _batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start+size]


@torch.no_grad()
def values_for(network, states, pools, size=128):
    result = []
    for start in range(0, len(states), size):
        _, values = network(states[start:start+size], pools[start:start+size])
        result.extend(values.detach().cpu().tolist())
    return np.asarray(result, dtype=np.float64)


def vector_gae(rows, next_values, lam=.95):
    """Separate interleaved environment streams; only native done cuts GAE."""
    advantages = np.zeros(len(rows), dtype=np.float64)
    following = {}
    for i in reversed(range(len(rows))):
        row = rows[i]
        continuation = 0.0 if row['done'] else 1.0
        residual = row['reward'] + continuation*next_values[i] - row['value']
        advantages[i] = residual + continuation*lam*following.get(row['env'], 0.0)
        following[row['env']] = advantages[i]
    return advantages, advantages + np.asarray([r['value'] for r in rows])


def ppo_update(network, optimizer, rows, config, fraction, teacher=()):
    device = next(network.parameters()).device
    next_values = values_for(network, [r['next_state'] for r in rows],
                             [r['next_candidates'] for r in rows], config['batch_size'])
    advantages, returns = vector_gae(rows, next_values, config['gae_lambda'])
    advantages = (advantages-advantages.mean())/max(float(advantages.std()), 1e-8)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    targets = torch.as_tensor(returns, dtype=torch.float32, device=device)
    old_log = torch.tensor([r['log_prob'] for r in rows], device=device)
    entropy_coef = config['entropy_start']*(1-fraction) + config['entropy_end']*fraction
    records, stopped = [], False
    for epoch in range(config['epochs']):
        for indices in _batches(np.random.permutation(len(rows)).tolist(), config['batch_size']):
            batch = [rows[i] for i in indices]
            scores, values = network([r['state'] for r in batch], [r['candidates'] for r in batch])
            distribution = Categorical(logits=scores)
            logp = distribution.log_prob(torch.tensor([r['action'] for r in batch], device=device))
            difference = logp-old_log[indices]
            ratio = difference.exp()
            kl = float(((ratio-1)-difference).mean().detach())
            if kl > config['target_kl'] and records:
                stopped = True
                break
            clipped = ratio.clamp(1-config['clip'], 1+config['clip'])
            actor_loss = -torch.minimum(ratio*advantages[indices], clipped*advantages[indices]).mean()
            value_loss = F.mse_loss(values, targets[indices])
            entropy = distribution.entropy().mean()
            loss = actor_loss + config['value_coef']*value_loss - entropy_coef*entropy
            auxiliary = scores.sum()*0.0
            if teacher and fraction < .7:
                sample = [teacher[i] for i in np.random.choice(len(teacher), min(16, len(teacher)), replace=False)]
                auxiliary = teacher_loss(network, sample)
                loss = loss + config['teacher_aux_coef']*(1-fraction/.7)*auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite candidate PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config['max_gradient_norm'], error_if_nonfinite=True)
            optimizer.step()
            records.append(dict(loss=float(loss.detach()), actor_loss=float(actor_loss.detach()),
                                value_loss=float(value_loss.detach()), entropy=float(entropy.detach()),
                                approx_kl=kl, clip_fraction=float(((ratio-1).abs()>config['clip']).float().mean().detach()),
                                gradient_norm=float(norm), teacher_loss=float(auxiliary.detach())))
        if stopped:
            break
    result = {k: float(np.mean([r[k] for r in records])) for k in records[0]} if records else {}
    return dict(result, optimizer_steps=len(records), kl_early_stop=stopped, entropy_coefficient=entropy_coef,
                rollout_events=len(rows), reward_mean=float(np.mean([r['reward'] for r in rows])))


def teacher_loss(network, rows):
    scores, _ = network([r['state'] for r in rows], [r['candidates'] for r in rows])
    labels = torch.tensor([r['label'] for r in rows], device=scores.device)
    return F.cross_entropy(scores, labels)


def imitation_update(network, optimizer, rows, config):
    loss = teacher_loss(network, rows)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), config['max_gradient_norm'], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def double_q_update(network, target, optimizer, rows, config):
    device = next(network.parameters()).device
    states, pools = [r['state'] for r in rows], [r['candidates'] for r in rows]
    scores, _ = network(states, pools)
    actions = torch.tensor([r['action'] for r in rows], device=device)
    selected = scores.gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_states, next_pools = [r['next_state'] for r in rows], [r['next_candidates'] for r in rows]
        online, _ = network(next_states, next_pools)
        delayed, _ = target(next_states, next_pools)
        best = online.argmax(1)
        bootstrap = delayed.gather(1, best[:, None]).squeeze(1)
        live = torch.tensor([not r['done'] for r in rows], device=device, dtype=torch.float32)
        rewards = torch.tensor([r['reward'] for r in rows], device=device)
        # gamma=1; a true terminal never bootstraps, including the 50-step end.
        targets = rewards + live*bootstrap
    loss = F.smooth_l1_loss(selected, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite Double DQN loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config['max_gradient_norm'], error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), q_mean=float(selected.mean().detach()),
                target_mean=float(targets.mean()), gradient_norm=float(norm), optimizer_steps=1)
