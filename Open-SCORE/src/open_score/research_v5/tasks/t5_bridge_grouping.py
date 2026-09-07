"""BRIDGE's merge/STOP construction adapted to a frozen rule executor.

Only group partitions are learned. At each physical event the common rule fixes
all target assignments and reserves. Internal DQN time is not physical time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from open_score.grouping.domain import DecisionState, Group, Grouping
from open_score.grouping.storage import (atomic_checkpoint, atomic_json, fingerprint,
    random_state, restore_random_state, seed_everything)
from open_score.research_v4.actions import partition_key, rule_grouping
from open_score.research_v5.protocol import stable_seed

VALIDATION_FRACTIONS = (.25, .5, .75, 1.)


def due_validation_fractions(completed, total, validated):
    return [fraction for fraction in VALIDATION_FRACTIONS
            if fraction not in validated and completed >= max(1, int(np.ceil(total*fraction)))]


def singleton_partition(rule):
    return Grouping(tuple(Group(g.target, (identity,)) for g in rule.groups for identity in g.members), rule.reserve)


def partition_controls(state):
    rule = rule_grouping(state)
    target_members = {}
    for group in rule.groups:
        target_members.setdefault(group.target, []).extend(group.members)
    grand = Grouping(tuple(Group(target, tuple(members)) for target, members in target_members.items()), rule.reserve)
    return [rule, singleton_partition(rule), grand]


def merge_actions(partition):
    """STOP first gives a stable conservative tie break without a size reward."""
    return [None]+[(i, j) for i, j in combinations(range(len(partition.groups)), 2)
                   if partition.groups[i].target == partition.groups[j].target]


def merge_partition(partition, action):
    if action is None:
        return partition
    i, j = action
    if not 0 <= i < j < len(partition.groups) or partition.groups[i].target != partition.groups[j].target:
        raise ValueError('merge must join two distinct groups with the same target')
    first, second = partition.groups[i], partition.groups[j]
    merged = Group(first.target, first.members+second.members)
    return Grouping(tuple(g for k, g in enumerate(partition.groups) if k not in (i, j))+(merged,), partition.reserve)


def _relation(partition, identities, maximum):
    index = {identity: i for i, identity in enumerate(identities)}
    result = np.zeros((maximum, maximum), np.float32)
    for group in partition.groups:
        rows = [index[i] for i in group.members if i in index]
        result[np.ix_(rows, rows)] = 1.
    return result.ravel()


class PartitionQNetwork(nn.Module):
    """Padded state/partition encoder and action-conditional Q, max 32 Red/Blue.

Physical encoding is shared across all merge actions, avoiding one Transformer
pass per possible pair. Padding and this representation are explicit departures
from BRIDGE's original environment, not claims of permutation invariance.
"""
    def __init__(self, max_members=32, max_targets=2, hidden_dim=256):
        super().__init__()
        self.max_members, self.max_targets, self.hidden_dim = int(max_members), int(max_targets), int(hidden_dim)
        width = 2*self.max_members*8+self.max_targets*8+4+self.max_members*(self.max_targets+1)+2*self.max_members**2
        self.encoder = nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.action_head = nn.Sequential(nn.Linear(hidden_dim+2*self.max_members+1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    @property
    def config(self):
        return {'max_members': self.max_members, 'max_targets': self.max_targets, 'hidden_dim': self.hidden_dim}

    def _state_features(self, state, partition):
        if max(len(state.ids('red')), len(state.ids('blue'))) > self.max_members or len(state.ids('targets')) > self.max_targets:
            raise ValueError('BRIDGE adaptation supports at most 32 members per side and two targets')
        partition.validate(state.ids('red'), state.ids('targets'), max_members=None)
        records = []
        for side, maximum in [('red', self.max_members), ('blue', self.max_members), ('targets', self.max_targets)]:
            values = np.zeros((maximum, 8), np.float32)
            for index, entity in enumerate(sorted(state.alive(side), key=lambda e: e.id)):
                values[index] = [1., *[p/2500. for p in entity.position], *[v/500. for v in entity.velocity], entity.health]
            records.append(values.ravel())
        ids = sorted(state.ids('red')); targets = sorted(state.ids('targets'))
        assignment = partition.assignment(); assigned = np.zeros((self.max_members, self.max_targets+1), np.float32)
        for i, identity in enumerate(ids):
            target = assignment[identity]
            assigned[i, self.max_targets if target is None else targets.index(target)] = 1.
        records.extend((np.asarray([state.step/state.max_steps, 1.-state.step/state.max_steps,
            len(ids)/self.max_members, len(state.ids('blue'))/self.max_members], np.float32), assigned.ravel(),
            _relation(state.previous, ids, self.max_members), _relation(partition, ids, self.max_members)))
        return np.concatenate(records)

    def _action_features(self, state, partition, actions):
        index = {identity: k for k, identity in enumerate(sorted(state.ids('red')))}
        result = np.zeros((len(actions), 2*self.max_members+1), np.float32)
        membership = np.zeros((len(partition.groups), self.max_members), np.float32)
        for group_index, group in enumerate(partition.groups):
            membership[group_index, [index[identity] for identity in group.members]] = 1.
        targets = np.asarray([group.target for group in partition.groups])
        nonterminal_rows, pairs = [], []
        for row, action in enumerate(actions):
            if action is None:
                result[row, -1] = 1.
            else:
                nonterminal_rows.append(row); pairs.append(action)
        if pairs:
            indices = np.asarray(pairs, dtype=np.int64)
            left, right = indices[:, 0], indices[:, 1]
            if np.any(left < 0) or np.any(left >= right) or np.any(right >= len(partition.groups)):
                raise ValueError('malformed merge action')
            if np.any(targets[left] != targets[right]):
                raise ValueError('cross-target merge is illegal')
            result[nonterminal_rows, :self.max_members] = membership[left]
            result[nonterminal_rows, self.max_members:2*self.max_members] = membership[right]
        return result

    def forward(self, states, partitions, action_batches):
        if not states or len(states) != len(partitions) or len(states) != len(action_batches) or any(not a for a in action_batches):
            raise ValueError('matching nonempty state, partition and action batches required')
        device = next(self.parameters()).device
        features = np.stack([self._state_features(s, p) for s, p in zip(states, partitions)])
        encoded = self.encoder(torch.as_tensor(features, device=device))
        counts = [len(actions) for actions in action_batches]
        owner = torch.repeat_interleave(torch.arange(len(states), device=device), torch.tensor(counts, device=device))
        action_values = np.concatenate([self._action_features(s, p, a) for s, p, a in zip(states, partitions, action_batches)])
        flat = self.action_head(torch.cat((encoded[owner], torch.as_tensor(action_values, device=device)), -1)).squeeze(-1)
        result = flat.new_full((len(states), max(counts)), -torch.inf)
        start = 0
        for i, count in enumerate(counts):
            result[i, :count] = flat[start:start+count]; start += count
        return result


def dqn_update(network, target, optimizer, batch, *, max_gradient_norm=40.):
    states = [row['state'] for row in batch]; partitions = [row['partition'] for row in batch]
    prediction = network(states, partitions, [[row['action']] for row in batch])[:, 0]
    rewards = prediction.new_tensor([row['reward'] for row in batch])
    bootstrap = torch.zeros_like(rewards)
    nonterminal = [i for i, row in enumerate(batch) if not row['done']]
    with torch.no_grad():
        if nonterminal:
            next_states = [states[i] for i in nonterminal]
            next_partitions = [batch[i]['following'] for i in nonterminal]
            next_actions = [merge_actions(p) for p in next_partitions]
            values = target(next_states, next_partitions, next_actions).max(1).values
            bootstrap[nonterminal] = values
    target_value = rewards+bootstrap  # internal gamma = 1, separate from physical events
    loss = F.mse_loss(prediction, target_value)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), float(max_gradient_norm))
    optimizer.step()
    return {'td_loss': float(loss.detach()), 'td_error': float((prediction.detach()-target_value).abs().mean()),
            'gradient_norm': float(norm)}


class MergePolicy:
    def __init__(self, network, device='cpu'):
        self.network = network.to(device).eval(); self.last_trace = {}

    @torch.no_grad()
    def act(self, state):
        rule = rule_grouping(state); partition = singleton_partition(rule)
        merges = 0; stopped = False; choices = []
        maximum = len(state.ids('red'))-len(rule.reserve)-len({g.target for g in rule.groups})
        for _ in range(maximum+1):
            actions = merge_actions(partition)
            if len(actions) == 1:
                break
            values = self.network([state], [partition], [actions])[0, :len(actions)]
            index = int(values.argmax()); action = actions[index]
            choices.append({'groups': len(partition.groups), 'action': action, 'q': float(values[index])})
            if action is None:
                stopped = True; break
            partition = merge_partition(partition, action); merges += 1
        if partition.assignment() != rule.assignment():
            raise AssertionError('merge policy changed fixed rule target assignments')
        self.last_trace = {'method': 'BRIDGE-FixedRule-Grouping', 'merge_count': merges,
            'stopped': stopped, 'stop_position': merges if stopped else None,
            'initial_groups': len(state.ids('red'))-len(rule.reserve),
            'final_groups': len(partition.groups), 'construction': choices,
            'rule': rule.to_dict(), 'target_assignment_fixed': True}
        return partition


def load_policy(path, device='cpu', **opts):
    saved = torch.load(Path(path), map_location=device, weights_only=False)
    network = PartitionQNetwork(**saved['model_config']); network.load_state_dict(saved['model'])
    return MergePolicy(network, device=device)


def _state_value(ctx, record, partition, cache, branch_seeds, path):
    from open_score.research_v5.simulator import paired_rollouts
    key = fingerprint(partition.to_dict())
    if key not in cache:
        started = time.monotonic()
        row = paired_rollouts(record['snapshot'], [partition], branch_seeds=branch_seeds)[0]
        cache[key] = {'action': partition.to_dict(), 'y': row['y'], 'outcomes': row['outcomes'],
            'physical_steps': row['physical_steps'], 'branch_seeds': branch_seeds,
            'branches': row.get('branches', []), 'terminal_steps': row.get('terminal_steps', []),
            'executor_version': ctx.config['executor'], 'opponent': ctx.config['opponent'],
            'continuation_version': 'rule_grouping_v1', 'protocol_hash': ctx.identity.get('protocol_hash'),
            'wall_s': time.monotonic()-started}
        # Persist once per (state, partition, branch set), independently of DQN checkpoints.
        ctx.store.put(path, cache)
        ctx.log('candidates', {'task': 'T5', 'family_id': record['family_id'],
            'state_id': record['state_id'], 'candidate_id': key, 'action': partition.to_dict(),
            'y': row['y'], 'branches': len(branch_seeds), 'purpose': 'fixed_branch_potential'})
        for i, seed in enumerate(branch_seeds):
            ctx.log('branches', {'task': 'T5', 'family_id': record['family_id'],
                'state_id': record['state_id'], 'candidate_id': key, 'branch_seed': seed,
                'outcome': row['outcomes'][i], 'physical_steps': row['physical_steps'][i],
                **(row['branches'][i] if row.get('branches') else {}),
                'purpose': 'fixed_branch_potential'})
    elif cache[key]['branch_seeds'] != branch_seeds:
        raise ValueError('potential cache branch set changed')
    return float(cache[key]['y'])


def run(ctx):
    cfg = ctx.config['t5']; output = Path(ctx.output); output.mkdir(parents=True, exist_ok=True)
    phase_path = 'phase_costs'
    costs = ctx.store.get(phase_path, {})
    def phase(name, **values):
        costs[name] = values
        ctx.store.put(phase_path, costs); ctx.log('phase_costs', {'phase': name, **values})
    seed_everything(ctx.seed)
    collection_started = time.monotonic()
    records = ctx.collect_states(int(cfg['states']), split='train', namespace='t5-construction')
    if not records:
        raise ValueError('T5 requires physical construction snapshots')
    collection_steps = sum(int(record.get('collection_physical_steps', 0)) for record in records)
    phase('state_collection', wall_s=costs.get('state_collection', {}).get('wall_s', 0.)+time.monotonic()-collection_started,
          simulation_physical_steps=collection_steps, states=len(records))
    model_config = {'max_members': 32, 'max_targets': 2,
                    'hidden_dim': 256}
    network = PartitionQNetwork(**model_config).to(ctx.device)
    target = copy.deepcopy(network).eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=float(cfg['learning_rate']))
    replay = deque(maxlen=int(cfg['buffer_size']))
    construction_index = transitions = updates = warmup = 0; elapsed_before = 0.; optimization_seconds = 0.
    validated = []; validation_seconds = 0.; validation_episodes = 0
    identity = dict(config=cfg, seed=ctx.seed, model=model_config)
    checkpoint = ctx.model_path('resume.pt')
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=ctx.device, weights_only=False)
        if saved['model_config'] != model_config:
            raise ValueError('T5 model structure differs from the saved checkpoint')
        network.load_state_dict(saved['model']); target.load_state_dict(saved['target'])
        optimizer.load_state_dict(saved['optimizer']); replay.extend(saved['replay'])
        construction_index, transitions, updates, warmup = (saved[key] for key in
            ('construction_index', 'internal_transitions', 'optimizer_steps', 'warmup_transitions'))
        elapsed_before = saved['training_seconds']; restore_random_state(saved['rng'])
        optimization_seconds = saved.get('optimization_seconds', 0.)
        validated = saved.get('validated_fractions', [])
        validation_seconds = saved.get('validation_seconds', 0.)
        validation_episodes = saved.get('validation_episodes', 0)
    caches = {}; cache_paths = {}
    for i, record in enumerate(records):
        path = f'potential_cache/state_{i:06d}.json'
        cache_paths[i] = path
        caches[i] = ctx.store.get(path, {})
    started = time.monotonic(); validation_seconds_before = validation_seconds; metrics = {}
    def validate_due(completed):
        nonlocal validation_seconds, validation_episodes
        due = due_validation_fractions(completed, int(cfg['episodes']), validated)
        if not due:
            return
        # Commit construction before validation; resume uses the same frozen
        # checkpoint and finishes only missing validation families.
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model_path = checkpoint
        ambient = random_state(); validation_started = time.monotonic()
        try:
            summary = ctx.evaluate(load_policy(model_path), 't5_bridge_grouping',
                                   split='validation', checkpoint=f'construction_{completed}')
        finally:
            restore_random_state(ambient)
        validation_seconds += time.monotonic()-validation_started
        validation_episodes += int(summary['episodes'])
        validated.extend(due)
        if summary['success_rate'] > ctx.store.get('best_validation_rate', -1.):
            atomic_checkpoint(ctx.model_path('best.pt'), payload)
            ctx.store.put('best_validation_rate', summary['success_rate'])
        payload.update(validated_fractions=list(validated), validation_seconds=validation_seconds,
                       validation_episodes=validation_episodes, rng=random_state())
        atomic_checkpoint(checkpoint, payload)
        ctx.log('validation', {'construction_episode': completed, 'fractions': due,
            'success_rate': summary['success_rate'], 'wins': summary['wins'], 'episodes': summary['episodes'],
            'checkpoint': f'construction_{completed}'})
        phase('validation', wall_s=validation_seconds, episodes=validation_episodes,
              checkpoints=len(validated))
    if construction_index:
        validate_due(construction_index)
    for episode in range(construction_index, int(cfg['episodes'])):
        index = episode % len(records); record = records[index]; state = record['state']
        rule = rule_grouping(state); partition = singleton_partition(rule)
        branch_seeds = [stable_seed(ctx.seed, 'T5', record['state_id'], 'potential', b)
                        for b in range(int(cfg['branches']))]
        initial = current = _state_value(ctx, record, partition, caches[index], branch_seeds, cache_paths[index])
        total_reward = 0.; merges = 0; stopped = False
        network.train()
        while True:
            actions = merge_actions(partition)
            epsilon = 1.-(1.-float(cfg['epsilon_end']))*min(1., transitions/int(cfg['epsilon_steps']))
            if len(actions) == 1:
                action = None
            elif random.random() < epsilon:
                action = random.choice(actions)
            else:
                with ctx.gpu(), torch.no_grad():
                    values = network([state], [partition], [actions])[0, :len(actions)]
                    action = actions[int(values.argmax())]
            following = merge_partition(partition, action)
            done = action is None or len(merge_actions(following)) == 1
            value = current if action is None else _state_value(ctx, record, following, caches[index], branch_seeds, cache_paths[index])
            reward = value-current
            replay.append({'state': state, 'partition': partition, 'action': action,
                           'following': following, 'reward': reward, 'done': done})
            transitions += 1; total_reward += reward
            if len(replay) >= int(cfg['batch_size']):
                optimization_started = time.monotonic()
                with ctx.gpu():
                    metrics = dqn_update(network, target, optimizer, random.sample(list(replay), int(cfg['batch_size'])),
                                         max_gradient_norm=float(cfg['max_gradient_norm']))
                optimization_seconds += time.monotonic()-optimization_started
                updates += 1
                if updates % int(cfg['target_interval']) == 0:
                    target.load_state_dict(network.state_dict())
            else:
                warmup += 1
            ctx.log('internal_transitions', {'construction_episode': episode, 'state_id': record['state_id'],
                'family_id': record['family_id'], 'internal_step': merges, 'action': action,
                'partition': partition.to_dict(), 'following': following.to_dict(), 'reward': reward,
                'potential_before': current, 'potential_after': value, 'internal_done': done,
                'physical_state_advanced': False, 'epsilon': epsilon, 'optimizer_steps': updates, **metrics})
            partition, current = following, value
            if action is not None: merges += 1
            else: stopped = True
            if done:
                break
        if abs(total_reward-(current-initial)) > 1e-7 or partition.assignment() != rule.assignment():
            raise AssertionError('BRIDGE potential/target assignment invariants failed')
        simulation_steps = collection_steps+sum(sum(row['physical_steps']) for cache in caches.values() for row in cache.values())
        elapsed = elapsed_before+time.monotonic()-started-(validation_seconds-validation_seconds_before)
        row = {'construction_episode': episode+1, 'episodes_target': int(cfg['episodes']),
            'internal_transitions': transitions, 'optimizer_steps': updates, 'epsilon': epsilon,
            'merge_count': merges, 'stopped': stopped, 'final_potential': current,
            'initial_potential': initial, 'return': total_reward, 'offline_sim_steps': simulation_steps,
            'training_seconds': elapsed, 'warmup_transitions': warmup, **metrics}
        # A complete construction is the recovery boundary: replay, optimizer,
        # target network and all RNG restore together; physical caches remain reusable.
        atomic_checkpoint(checkpoint, {'schema': 'v5-bridge-fixed-rule-v1', 'identity': identity,
            'model': network.state_dict(), 'target': target.state_dict(), 'model_config': network.config,
            'optimizer': optimizer.state_dict(), 'replay': list(replay), 'rng': random_state(),
            'construction_index': episode+1, 'internal_transitions': transitions,
            'optimizer_steps': updates, 'warmup_transitions': warmup, 'training_seconds': elapsed,
            'optimization_seconds': optimization_seconds,
            'validated_fractions': list(validated), 'validation_seconds': validation_seconds,
            'validation_episodes': validation_episodes,
            'offline_sim_steps': simulation_steps, 'metadata': {'protocol': ctx.config['version'],
                'opponent': ctx.config['opponent'], 'executor': ctx.config['executor'], 'seed': ctx.seed,
                'protocol_hash': ctx.identity.get('protocol_hash'), 'continuation': 'rule_grouping_v1'}})
        ctx.log('training', row)
        ctx.progress(phase='train', **row)
        phase('potential_simulation', wall_s=sum(row.get('wall_s', 0.) for cache in caches.values() for row in cache.values()),
              simulation_physical_steps=simulation_steps-collection_steps,
              candidate_rows=sum(len(cache) for cache in caches.values()), states=len(caches))
        phase('optimization', wall_s=optimization_seconds, optimizer_steps=updates,
              internal_transitions=transitions, construction_episodes=episode+1)
        validate_due(episode+1)
    training_seconds = elapsed_before+time.monotonic()-started-(validation_seconds-validation_seconds_before)
    phase('construction_overhead', wall_s=max(0., training_seconds-optimization_seconds-
          sum(row.get('wall_s', 0.) for cache in caches.values() for row in cache.values())),
          construction_episodes=int(cfg['episodes']))
    policy = MergePolicy(network.to('cpu'))
    ctx.progress(phase='evaluate')
    evaluation_started = time.monotonic()
    evaluation = ctx.evaluate(policy, 't5_bridge_grouping', checkpoint='final_construction')
    phase('evaluation', wall_s=costs.get('evaluation', {}).get('wall_s', 0.)+time.monotonic()-evaluation_started,
          episodes=evaluation['episodes'])
    final_values = [row['y'] for cache in caches.values() for row in cache.values()]
    result = {'task': 'T5', 'status': 'complete', 'adaptation_scope': 'rule_target_assignment_plus_learned_partition',
        'reproduction': 'paper_algorithm_reimplementation_of_merge_STOP_DQN_with_fixed_rule_lower',
        'training_seed': ctx.seed, 'construction_episodes': int(cfg['episodes']),
        'internal_transitions': transitions, 'optimizer_steps': updates, 'warmup_transitions': warmup,
        'offline_sim_steps': collection_steps+sum(sum(row['physical_steps']) for cache in caches.values() for row in cache.values()),
        'state_collection_physical_steps': collection_steps, 'real_train_steps': 0,
        'training_seconds': training_seconds, 'validation_seconds': validation_seconds,
        'validation_episodes': validation_episodes, 'validated_fractions': list(validated), 'evaluation': evaluation,
        'potential': {'cached_partitions': len(final_values),
            'zero_labels': sum(v == 0 for v in final_values),
            'non_tie_states': sum(len({r['y'] for r in cache.values()}) > 1 for cache in caches.values())}}
    ctx.store.put('result', result)
    checkpoint.replace(ctx.model_path('final.pt'))
    ctx.progress(phase='complete', optimizer_steps=updates)
    return result


