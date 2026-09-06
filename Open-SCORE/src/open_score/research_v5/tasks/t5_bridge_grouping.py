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
        atomic_json(path, cache)
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


def partition_diagnostic_report(output):
    """Read the independently verified common snapshots; do not resimulate."""
    records = []
    for path in sorted((Path(output)/'diagnostics/t5_bridge_grouping/diagnostic').glob('*.pt')):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        summary, rows = saved['summary'], saved['verification']
        state = DecisionState.from_dict(rows[0]['state'])
        rule, singletons, grand = partition_controls(state)
        by_key = {partition_key(Grouping.from_dict(row['action'])): row for row in rows}
        selected = rows[summary['selected_index']]
        selected_plan = Grouping.from_dict(selected['action'])
        if selected_plan.assignment() != rule.assignment():
            raise AssertionError('T5 diagnostic must fix target identity assignment')
        record = {'family_id': summary['family_id'], 'state_id': summary['state_id'],
                  'red_count': summary['red_count'], 'blue_count': summary['blue_count']}
        grand_row = by_key[partition_key(grand)]
        for name, row in [('rule', by_key[partition_key(rule)]),
                          ('singletons', by_key[partition_key(singletons)]),
                          ('grand', grand_row), ('learned', selected)]:
            record[name] = float(row['y'])
            record[name+'_minus_grand'] = float(np.mean(np.asarray(row['outcomes'])-grand_row['outcomes']))
            record[name+'_branch_vector_changed'] = row['outcomes'] != grand_row['outcomes']
        records.append(record)
    result = {'states': len(records), 'families': len({row['family_id'] for row in records}),
              'comparison': 'fixed_rule_target_identity_assignment; independent_verification_branches',
              'rows': records, 'directions_vs_grand': {}}
    for name in ('rule', 'singletons', 'learned'):
        differences = [row[name+'_minus_grand'] for row in records]
        result['directions_vs_grand'][name] = {'better_states': sum(value > 0 for value in differences),
            'worse_states': sum(value < 0 for value in differences), 'tie_states': sum(value == 0 for value in differences),
            'branch_vector_changed_states': sum(row[name+'_branch_vector_changed'] for row in records),
            'mean_gain': float(np.mean(differences)) if differences else None}
    atomic_json(Path(output)/'pure_partition_diagnostic.json', result)
    return result


def run(ctx):
    cfg = ctx.config['t5']; output = Path(ctx.output); output.mkdir(parents=True, exist_ok=True)
    phase_path = output/'phase_costs.json'
    costs = json.loads(phase_path.read_text(encoding='utf-8')) if phase_path.exists() else {}
    def phase(name, **values):
        costs[name] = values
        atomic_json(phase_path, costs); ctx.log('phase_costs', {'phase': name, **values})
    seed_everything(ctx.seed)
    collection_started = time.monotonic()
    records = ctx.collect_states(int(cfg['states']), split='train', namespace='t5-construction')
    if not records:
        raise ValueError('T5 requires physical construction snapshots')
    collection_steps = sum(int(record.get('collection_physical_steps', 0)) for record in records)
    phase('state_collection', wall_s=costs.get('state_collection', {}).get('wall_s', 0.)+time.monotonic()-collection_started,
          simulation_physical_steps=collection_steps, states=len(records))
    model_config = {'max_members': 32, 'max_targets': 2,
                    'hidden_dim': 64 if ctx.config.get('smoke') else 256}
    network = PartitionQNetwork(**model_config).to(ctx.device)
    target = copy.deepcopy(network).eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=float(cfg['learning_rate']))
    replay = deque(maxlen=int(cfg['buffer_size']))
    construction_index = transitions = updates = warmup = 0; elapsed_before = 0.; optimization_seconds = 0.
    validated = []; validation_seconds = 0.; validation_episodes = 0
    identity = fingerprint({'config': cfg, 'protocol': ctx.config['version'], 'seed': ctx.seed,
                            'states': [row['state_id'] for row in records], 'model': model_config})
    checkpoint = output/'latest.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=ctx.device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('T5 protocol/state/model changed; use a new output directory')
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
        path = output/'potential_cache'/f'state_{i:06d}.json'
        cache_paths[i] = path
        caches[i] = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    initial_path = output/'initialized.pt'
    if not initial_path.exists():
        atomic_checkpoint(initial_path, {'model': network.state_dict(), 'model_config': network.config,
                                        'schema': 'v5-bridge-fixed-rule-v1', 'identity': identity})
    started = time.monotonic(); validation_seconds_before = validation_seconds; metrics = {}
    def validate_due(completed):
        nonlocal validation_seconds, validation_episodes
        due = due_validation_fractions(completed, int(cfg['episodes']), validated)
        if not due:
            return
        # Commit construction before validation; resume uses the same frozen
        # checkpoint and finishes only missing validation families.
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model_path = output/f'construction_{completed}.pt'
        if not model_path.exists():
            atomic_checkpoint(model_path, payload)
        ambient = random_state(); validation_started = time.monotonic()
        try:
            summary = ctx.evaluate(load_policy(model_path), 't5_bridge_grouping',
                                   split='validation', checkpoint=f'construction_{completed}')
        finally:
            restore_random_state(ambient)
        validation_seconds += time.monotonic()-validation_started
        validation_episodes += int(summary['episodes'])
        validated.extend(due)
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
    diagnostic_started = time.monotonic()
    diagnostics = ctx.diagnose(policy, 't5_bridge_grouping', candidate_provider=partition_controls)
    phase('diagnostics', wall_s=costs.get('diagnostics', {}).get('wall_s', 0.)+time.monotonic()-diagnostic_started,
          states=sum(value['states'] for value in diagnostics.values()),
          simulation_physical_steps=sum(value['simulated_physical_steps'] for value in diagnostics.values()))
    pure_partition = partition_diagnostic_report(output)
    final_values = [row['y'] for cache in caches.values() for row in cache.values()]
    result = {'task': 'T5', 'status': 'complete', 'adaptation_scope': 'rule_target_assignment_plus_learned_partition',
        'reproduction': 'paper_algorithm_reimplementation_of_merge_STOP_DQN_with_fixed_rule_lower',
        'training_seed': ctx.seed, 'construction_episodes': int(cfg['episodes']),
        'internal_transitions': transitions, 'optimizer_steps': updates, 'warmup_transitions': warmup,
        'offline_sim_steps': collection_steps+sum(sum(row['physical_steps']) for cache in caches.values() for row in cache.values()),
        'state_collection_physical_steps': collection_steps, 'real_train_steps': 0,
        'training_seconds': training_seconds, 'validation_seconds': validation_seconds,
        'validation_episodes': validation_episodes, 'validated_fractions': list(validated), 'evaluation': evaluation,
        'diagnostics': diagnostics, 'pure_partition_diagnostic': pure_partition,
        'potential': {'cached_partitions': len(final_values),
            'zero_labels': sum(v == 0 for v in final_values),
            'non_tie_states': sum(len({r['y'] for r in cache.values()}) > 1 for cache in caches.values())}}
    atomic_json(output/'summary.json', result)
    learned_direction = pure_partition['directions_vs_grand']['learned']
    common = diagnostics['diagnostic']
    (output/'analysis.md').write_text('# T5 BRIDGE 固定规则分组适配\n\n'
        '每次指挥事件先按共同规则确定目标与身份归属，再从单体组开始学习同目标合并或停止。'
        '固定分支缓存终局势值，内部回报为势差、gamma=1；STOP 的奖励为零。'
        '没有训练下层，不属于原 BRIDGE 双层实验的完整复现。\n\n'
        f"已完成 {int(cfg['episodes'])} 次构造、{transitions} 条内部 transition、{updates} 次 DQN 更新。"
        f"正式独立测试 {evaluation['wins']}/{evaluation['episodes']}。\n\n"
        f"训练快照中有 {result['potential']['non_tie_states']}/{len(records)} 个曾观察到势值不同。"
        f"共同状态独立复核相对规则平均收益差为 {common['selected_vs_rule_verification_gain']:+.4f}。\n\n"
        f"固定目标与身份归属，相对大组：学习分区更好 {learned_direction['better_states']} 个状态，"
        f"更差 {learned_direction['worse_states']}，平局 {learned_direction['tie_states']}。"
        '逐分支变化与平均收益变化分开计数，见pure_partition_diagnostic.json。\n\n'
        '若势值多数平局，采样信息不足是待检验瓶颈；若教师独立复核更好而学习分区未改善，'
        '应检查构造策略学习和访问分布。以上为有限分支单种子证据，固定目标范围的负结果不能当完整BRIDGE的上限。\n',
        encoding='utf-8')
    ctx.progress(phase='complete', optimizer_steps=updates)
    return result


def benchmark(ctx):
    """Bounded production-shape sampler/update probe; never touches training data."""
    from open_score.research_v5.paired import benchmark_records
    from open_score.research_v5.simulator import paired_rollouts
    torch.set_num_threads(1)
    started = time.monotonic(); phases = {}
    phase_started = time.monotonic()
    iteration = getattr(ctx, '_t5_benchmark_iteration', 0)
    ctx._t5_benchmark_iteration = iteration+1
    records, collection_steps = benchmark_records(ctx, f'T5-calibration-only:{iteration}')
    phases['state_collection'] = {'wall_s': time.monotonic()-phase_started,
        'simulation_physical_steps': collection_steps, 'states': len(records),
        'units': len(records), 'unit': 'states', 'optimizer_steps': 0}
    phase_started = time.monotonic(); examples = []; physical_steps = candidate_rows = 0; construction_merges = []
    for record in records:
        state = record['state']; partition = singleton_partition(rule_grouping(state))
        seeds = [stable_seed(ctx.seed, record['state_id'], 'T5-benchmark-branch', i)
                 for i in range(int(ctx.config['t5']['branches']))]
        cache = {}; rng = random.Random(stable_seed(record['state_id'], 'construction'))
        def potential(plan):
            nonlocal physical_steps, candidate_rows
            key = partition_key(plan)
            if key not in cache:
                row = paired_rollouts(record['snapshot'], [plan], branch_seeds=seeds)[0]
                physical_steps += sum(row['physical_steps']); candidate_rows += 1
                cache[key] = row['y']
            return cache[key]
        current = potential(partition); merges = 0
        while True:
            actions = merge_actions(partition); action = rng.choice(actions)
            following = merge_partition(partition, action)
            value = current if action is None else potential(following)
            done = action is None or len(merge_actions(following)) == 1
            examples.append({'state': state, 'partition': partition, 'action': action, 'following': following,
                'reward': value-current, 'done': done})
            partition, current = following, value
            if action is not None: merges += 1
            if done: break
        construction_merges.append(merges)
    phases['potential_simulation'] = {'wall_s': time.monotonic()-phase_started,
        'simulation_physical_steps': physical_steps, 'candidate_rows': candidate_rows,
        'states': len(records), 'units': candidate_rows, 'unit': 'candidate_rows', 'optimizer_steps': 0,
        'construction_episodes': len(records), 'internal_transitions': len(examples),
        'mean_internal_transitions': len(examples)/len(records),
        'mean_unique_potentials': candidate_rows/len(records),
        'merge_counts': construction_merges, 'mean_merges': float(np.mean(construction_merges))}
    setup_started = time.monotonic()
    network = PartitionQNetwork(hidden_dim=256).to(ctx.device); target = copy.deepcopy(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=float(ctx.config['t5']['learning_rate']))
    phases['model_setup'] = {'wall_s': time.monotonic()-setup_started, 'units': 1, 'unit': 'models',
                            'parameters': sum(p.numel() for p in network.parameters()), 'optimizer_steps': 0}
    batch = [examples[i % len(examples)] for i in range(128)]
    phase_started = time.monotonic()
    for _ in range(2):
        with ctx.gpu():
            dqn_update(network, target, optimizer, batch, max_gradient_norm=40.)
            if str(ctx.device).startswith('cuda'):
                torch.cuda.synchronize()
    phases['optimization'] = {'wall_s': time.monotonic()-phase_started, 'optimizer_steps': 2,
        'batches': 2, 'batch_size': 128, 'units': 2, 'unit': 'optimizer_steps',
        'simulation_physical_steps': 0}
    from open_score.research_v5.evaluate import episode as evaluate_episode
    policy = MergePolicy(network.to('cpu'))
    phase_started = time.monotonic(); evaluation_rows = []
    for index in (0, len(ctx.config['cells'])//2, len(ctx.config['cells'])-1):
        row, _ = evaluate_episode(policy, ctx.spec(index, 'calibration', f'T5-evaluation:{iteration}'))
        evaluation_rows.append(row)
    phases['evaluation'] = {'wall_s': time.monotonic()-phase_started, 'episodes': len(evaluation_rows),
        'units': len(evaluation_rows), 'unit': 'episodes',
        'real_physical_steps': sum(row['physical_steps'] for row in evaluation_rows),
        'command_events': sum(row['command_events'] for row in evaluation_rows), 'optimizer_steps': 0}
    return {'task_id': 'T5', 'phases': phases, 'wall_s': time.monotonic()-started,
            'state_steps': [row['state'].step for row in records],
            'gpu_wait_s': getattr(ctx, '_gpu_wait_s', 0.)}
