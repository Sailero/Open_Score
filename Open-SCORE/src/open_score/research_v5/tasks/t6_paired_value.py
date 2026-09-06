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
    folder = Path(ctx.output)/'data'/split
    folder.mkdir(parents=True, exist_ok=True)
    all_rows = []; simulation_steps = 0
    for index, record in enumerate(records):
        path = folder/f'state_{index:06d}.json'
        state = record['state']
        identity = fingerprint({'state': state.to_dict(), 'family_id': record['family_id'],
            'state_id': record['state_id'], 'config': cfg, 'seed': ctx.seed, 'split': split})
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved['identity'] != identity:
                raise ValueError('T6 state collection changed; use a new run directory')
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
            atomic_json(path, {'identity': identity, 'rows': rows})
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
    old_manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8')) if (folder/'manifest.json').exists() else {}
    atomic_json(folder/'manifest.json', {'status': 'complete', 'states': len(groups),
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
    phase_path = output/'phase_costs.json'
    costs = json.loads(phase_path.read_text(encoding='utf-8')) if phase_path.exists() else {}
    def phase(name, **values):
        costs[name] = values
        atomic_json(phase_path, costs); ctx.log('phase_costs', {'phase': name, **values})
    train, train_steps = collect_dataset(ctx, 'train', int(cfg['train_states']))
    validation, validation_steps = collect_dataset(ctx, 'validation', int(cfg['validation_states']))
    for split in ('train', 'validation'):
        manifest = json.loads((output/'data'/split/'manifest.json').read_text(encoding='utf-8'))
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
        model, result = train_model(train, validation, output/kind, kind=kind, seed=ctx.seed,
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
    atomic_json(output/'gate_selection.json', {'selection': 'validation_success_ties_larger_threshold',
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
        diagnostic_started = time.monotonic()
        diagnostics[name] = ctx.diagnose(policy, name, include_own=name == 't6_adv')
        phase('diagnostics_'+name, wall_s=costs.get('diagnostics_'+name, {}).get('wall_s', 0.)+time.monotonic()-diagnostic_started,
              states=sum(row['states'] for row in diagnostics[name].values()),
              simulation_physical_steps=sum(row['simulated_physical_steps'] for row in diagnostics[name].values()))
    comparison = {}
    for kind in ('bce', 'adv'):
        comparison[kind] = audit_model(models[kind], validation, kind)
    atomic_json(output/'paired_validation_audit.json', comparison)
    result = {'task': 'T6', 'status': 'complete',
        'adaptation_scope': 'same_data_absolute_BCE_vs_paired_advantage_with_empirical_rule_gate',
        'training': training_results, 'offline_sim_steps': train_steps+validation_steps,
        'gate': chosen['threshold'], 'evaluation': results, 'diagnostics': diagnostics,
        'calibration_ranking': comparison,
        'interpretation': 'One training seed; validation gate is not a guaranteed policy improvement bound.'}
    atomic_json(output/'summary.json', result)
    training_labels = label_statistics(train)
    adv_minus_bce = _success(results['t6_adv'])-_success(results['t6_bce'])
    (output/'analysis.md').write_text('# T6 配对收益监督诊断\n\n'
        'BCE 与 ADV 使用完全相同的新终局数据、初始化、编码容量与完整状态训练批次。'
        '规则候选计入共同搜索预算，严格正收益才替换规则。门槛由共同验证开局选择，'
        '没有用最终测试挑门槛，也不声称统计安全保证。\n\n'
        + '\n'.join(f"- {name}: 成功率 {_success(value):.3%}" for name, value in results.items())
        + f"\n\n所选 ADV 门槛：{chosen['threshold']}。原始分支、状态平局与排序见同目录 JSON。\n"
        + f"训练状态全零 {training_labels['all_zero_states']}/{training_labels['states']}，"
        + f"非平局 {training_labels['non_tie_states']}/{training_labels['states']}。"
        + f"独立测试 ADV−BCE 点估计 {adv_minus_bce:+.3%}；这不是跨训练种子稳定性结论。\n\n"
        + f"共同状态独立复核相对规则：BCE {diagnostics['t6_bce']['diagnostic']['selected_vs_rule_verification_gain']:+.4f}，"
        + f"ADV {diagnostics['t6_adv']['diagnostic']['selected_vs_rule_verification_gain']:+.4f}。\n\n"
        + '同数据两臂都改善时，不能把全部收益归因于差分损失；门槛只减少有害替换时，不能称为学会更优方案。'
        '部署率改善不能单独证明任务有效；应同时检查配对成功差、候选复核和连续重规划结果。\n',
        encoding='utf-8')
    ctx.progress(phase='complete', offline_sim_steps=train_steps+validation_steps)
    return result


def benchmark(ctx):
    """Production-size state-batch work, independent of formal fit/evaluation."""
    import torch
    from open_score.research_v4.outcomes import GlobalOutcomeNetwork
    from open_score.research_v5.paired import benchmark_records, state_balanced_loss
    from open_score.research_v5.simulator import paired_rollouts
    torch.set_num_threads(1)
    started = time.monotonic(); phases = {}; cfg = ctx.config['t6']
    phase_started = time.monotonic()
    iteration = getattr(ctx, '_t6_benchmark_iteration', 0)
    ctx._t6_benchmark_iteration = iteration+1
    records, collection_steps = benchmark_records(ctx, f'T6-calibration-only:{iteration}')
    phases['state_collection'] = {'wall_s': time.monotonic()-phase_started,
        'simulation_physical_steps': collection_steps, 'states': len(records),
        'units': len(records), 'unit': 'states', 'optimizer_steps': 0}
    phase_started = time.monotonic(); all_rows = []; physical_steps = 0
    for record in records:
        candidates = candidate_pool(record['state'], budget=int(cfg['candidates']),
            rng=np.random.default_rng(stable_seed(ctx.seed, record['state_id'], 'candidate')))
        seeds = [stable_seed(ctx.seed, record['state_id'], 'T6-benchmark-label', i)
                 for i in range(int(cfg['branches']))]
        rows = paired_rollouts(record['snapshot'], candidates, branch_seeds=seeds)
        for row in rows:
            row.update(family_id=record['family_id'], state_id=record['state_id'], episode_spec=record['episode_spec'])
        physical_steps += sum(sum(row['physical_steps']) for row in rows)
        all_rows.extend(rows)
    groups = group_rows(all_rows)
    phases['label_collection'] = {'wall_s': time.monotonic()-phase_started,
        'simulation_physical_steps': physical_steps, 'candidate_rows': len(all_rows),
        'states': len(groups), 'units': len(groups), 'unit': 'states', 'optimizer_steps': 0}
    for kind in ('bce', 'adv'):
        model = GlobalOutcomeNetwork(hidden_dim=128, heads=4, layers=2).to(ctx.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['learning_rate']))
        batch = [groups[i % len(groups)] for i in range(16)]
        phase_started = time.monotonic()
        with ctx.gpu():
            optimizer.zero_grad(set_to_none=True)
            loss = state_balanced_loss(model, batch, kind); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.); optimizer.step()
            if str(ctx.device).startswith('cuda'):
                torch.cuda.synchronize()
        phases['train_'+kind] = {'wall_s': time.monotonic()-phase_started,
            'optimizer_steps': 1, 'batches': 1, 'batch_size': 16, 'states': 16,
            'units': 1, 'unit': 'optimizer_steps', 'simulation_physical_steps': 0,
            'parameters': sum(p.numel() for p in model.parameters())}
    from open_score.research_v5.evaluate import episode as evaluate_episode
    policy = PairedPolicy(model.to('cpu'), 'adv', budget=int(cfg['search_budget']))
    phase_started = time.monotonic(); evaluation_rows = []
    for index in (0, len(ctx.config['cells'])//2, len(ctx.config['cells'])-1):
        row, _ = evaluate_episode(policy, ctx.spec(index, 'calibration', f'T6-evaluation:{iteration}'))
        evaluation_rows.append(row)
    phases['evaluation'] = {'wall_s': time.monotonic()-phase_started, 'episodes': len(evaluation_rows),
        'units': len(evaluation_rows), 'unit': 'episodes',
        'real_physical_steps': sum(row['physical_steps'] for row in evaluation_rows),
        'command_events': sum(row['command_events'] for row in evaluation_rows), 'optimizer_steps': 0,
        'search_budget': int(cfg['search_budget']), 'kind': 'adv'}
    return {'task_id': 'T6', 'phases': phases, 'wall_s': time.monotonic()-started,
            'state_steps': [row['state'].step for row in records],
            'batch_note': '16 state slots sampled with replacement from three independent calibration families',
            'gpu_wait_s': getattr(ctx, '_gpu_wait_s', 0.)}
