"""T6: same-data BCE/paired advantage, plus a validation-selected gate."""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np

from open_score.grouping.storage import atomic_json, fingerprint
from open_score.grouping.domain import Grouping
from open_score.research_v4.actions import candidate_pool, partition_key, rule_grouping
from open_score.research_v5.paired import (PairedPolicy, audit_model, group_rows,
    label_statistics, load_policy, train_model)
from open_score.research_v5.protocol import stable_seed


def collect_dataset(ctx, split, count):
    from open_score.research_v5.simulator import paired_rollouts
    cfg = ctx.config['t6']
    started = time.monotonic()
    records = ctx.collect_states(count, split=split, namespace='t6-paired')
    folder = f'data/{split}'
    all_rows = []; simulation_steps = 0
    for index, record in enumerate(records):
        path = f'{folder}/state_{index:06d}.json'
        state = record['state']
        saved = ctx.store.get(path)
        if saved is not None:
            rows = saved['rows']
        else:
            rng = np.random.default_rng(stable_seed(ctx.seed, 'T6', split, record['state_id'], 'candidate'))
            candidates = candidate_pool(state, budget=int(cfg['candidates']), rng=rng)
            branch_seeds = [stable_seed(ctx.seed, 'T6', split, record['state_id'], 'label', b)
                            for b in range(int(cfg['branches']))]
            rows = paired_rollouts(record['snapshot'], candidates, branch_seeds=branch_seeds)
            for row in rows:
                row.update(family_id=record['family_id'], state_id=record['state_id'],
                           split=split, episode_spec=record['episode_spec'],
                           executor_version=ctx.config['executor'], opponent=ctx.config['opponent'],
                           protocol_hash=ctx.identity.get('protocol_hash'))
            ctx.store.put(path, {'rows': rows})
            rule_key = partition_key(rule_grouping(state))
            anchor = next(row for row in rows if partition_key(Grouping.from_dict(row['action'])) == rule_key)
            for row in rows:
                candidate_id = fingerprint(row['action'])
                ctx.log('candidates', {'task': 'T6', 'split': split,
                    'family_id': record['family_id'], 'state_id': record['state_id'],
                    'candidate_id': candidate_id, 'action': row['action'], 'y': row['y'],
                    'paired_advantage': float(np.mean(np.asarray(row['outcomes'])-anchor['outcomes'])),
                    'branches': len(branch_seeds)})
                for b, branch_seed in enumerate(branch_seeds):
                    ctx.log('branches', {'task': 'T6', 'split': split,
                        'family_id': record['family_id'], 'state_id': record['state_id'],
                        'candidate_id': candidate_id, 'branch_seed': branch_seed,
                        'outcome': row['outcomes'][b], 'rule_outcome': anchor['outcomes'][b],
                        'difference': row['outcomes'][b]-anchor['outcomes'][b],
                        'physical_steps': row['physical_steps'][b],
                        **(row['branches'][b] if row.get('branches') else {}),
                        'purpose': 'supervised_label'})
        all_rows.extend(rows)
        simulation_steps += sum(sum(row['physical_steps']) for row in rows)
        ctx.progress(phase='collect_'+split, states=index+1, states_target=len(records),
                     offline_sim_steps=simulation_steps)
    groups = group_rows(all_rows)
    collection_steps = sum(int(row.get('collection_physical_steps', 0)) for row in records)
    old_manifest = ctx.store.get(folder+'/manifest.json', {})
    ctx.store.put(folder+'/manifest.json', {'status': 'complete', 'states': len(groups),
        'offline_sim_steps': simulation_steps+collection_steps, 'label_sim_steps': simulation_steps,
        'state_collection_physical_steps': collection_steps, 'labels': label_statistics(groups),
        'wall_s': old_manifest.get('wall_s', 0.)+time.monotonic()-started})
    return groups, simulation_steps+collection_steps


def _success(summary):
    for key in ('success_rate', 'win_rate', 'overall_success'):
        if key in summary and summary[key] is not None:
            return float(summary[key])
    wins = summary.get('wins', summary.get('successes'))
    episodes = summary.get('episodes', summary.get('count'))
    if wins is not None and episodes:
        return float(wins)/int(episodes)
    raise ValueError('online validation summary must contain an actual success rate')


def run(ctx):
    cfg = ctx.config['t6']; output = Path(ctx.output)
    phase_path = 'phase_costs'
    costs = ctx.store.get(phase_path, {})
    def phase(name, **values):
        costs[name] = values
        ctx.store.put(phase_path, costs); ctx.log('phase_costs', {'phase': name, **values})
    train, train_steps = collect_dataset(ctx, 'train', int(cfg['train_states']))
    validation, validation_steps = collect_dataset(ctx, 'validation', int(cfg['validation_states']))
    for split in ('train', 'validation'):
        manifest = ctx.store.get(f'data/{split}/manifest.json')
        phase('collect_'+split, wall_s=manifest['wall_s'], simulation_physical_steps=manifest['offline_sim_steps'],
              states=manifest['states'], candidate_rows=manifest['labels']['candidates'])
    training_results = {}; models = {}
    for kind in ('bce', 'adv'):
        ctx.progress(phase='train_'+kind, epochs_target=int(cfg['epochs']))
        def progress(row):
            ctx.progress(phase='train_'+kind, epoch=row['epoch'], epochs_target=row['epochs'],
                         loss=row['loss'], optimizer_steps=row['updates'],
                         validation_ranking_accuracy=row['validation']['ranking_accuracy'])
            ctx.log('training', row)
        model, result = train_model(train, validation, output, kind=kind, seed=ctx.seed,
            epochs=int(cfg['epochs']), batch_states=int(cfg['batch_size']), device=ctx.device,
            learning_rate=float(cfg['learning_rate']), model_config=ctx.config['model'],
            gpu=ctx.gpu, progress=progress, metadata={'protocol': ctx.config['version'],
                'seed': ctx.seed, 'executor': ctx.config['executor'], 'opponent': ctx.config['opponent'],
                'protocol_hash': ctx.identity.get('protocol_hash'), 'continuation': 'rule_grouping_v1'})
        models[kind] = model.to('cpu').eval(); training_results[kind] = result
        phase('train_'+kind, wall_s=result['training_seconds'], optimizer_steps=result['updates'],
              states=len(train), epochs=int(cfg['epochs']))
    gate_results = []
    gate_started = time.monotonic()
    for threshold in cfg['thresholds']:
        policy = PairedPolicy(models['adv'], 'adv', budget=int(cfg['search_budget']), threshold=float(threshold))
        name = 't6_adv_gate_'+str(threshold).replace('.', 'p')
        ctx.progress(phase='gate_validation', threshold=float(threshold))
        summary = ctx.evaluate(policy, name, split='validation', checkpoint='final_epoch')
        gate_results.append({'threshold': float(threshold), 'success_rate': _success(summary), 'summary': summary})
    phase('gate_validation', wall_s=costs.get('gate_validation', {}).get('wall_s', 0.)+time.monotonic()-gate_started,
          episodes=sum(row['summary']['episodes'] for row in gate_results))
    chosen = max(gate_results, key=lambda row: (row['success_rate'], row['threshold']))
    ctx.store.put('gate_selection', {'selection': 'validation_success_ties_larger_threshold',
        'thresholds': gate_results, 'chosen_threshold': chosen['threshold'],
        'statistical_confidence_bound': False})
    results = {}; diagnostics = {}
    for name, kind, threshold in [('t6_bce', 'bce', 0.), ('t6_adv', 'adv', 0.),
                                 ('t6_adv_gated', 'adv', chosen['threshold'])]:
        policy = PairedPolicy(models[kind], kind, budget=int(cfg['search_budget']), threshold=threshold)
        ctx.progress(phase='evaluate_'+name)
        evaluation_started = time.monotonic()
        results[name] = ctx.evaluate(policy, name, checkpoint='final_epoch')
        phase('evaluation_'+name, wall_s=costs.get('evaluation_'+name, {}).get('wall_s', 0.)+time.monotonic()-evaluation_started,
              episodes=results[name]['episodes'])
    comparison = {}
    for kind in ('bce', 'adv'):
        comparison[kind] = audit_model(models[kind], validation, kind)
    ctx.store.put('paired_validation', comparison)
    result = {'task': 'T6', 'status': 'complete',
        'adaptation_scope': 'same_data_absolute_BCE_vs_paired_advantage_with_empirical_rule_gate',
        'training': training_results, 'offline_sim_steps': train_steps+validation_steps,
        'gate': chosen['threshold'], 'evaluation': results, 'diagnostics': diagnostics,
        'calibration_ranking': comparison,
        'interpretation': 'One training seed; validation gate is not a guaranteed policy improvement bound.'}
    ctx.store.put('result', result)
    ctx.progress(phase='complete', offline_sim_steps=train_steps+validation_steps)
    return result


