"""Predeclared independent verification; diagnostic families never train models."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import torch

from open_score.research_v4.actions import rule_grouping, partition_key
from open_score.research_v4.runner import atomic_json
from .planning import propose_plans, choose_with_rule_ties
from .protocol import stable_seed
from .simulator import env_from_snapshot, choose, paired_rollouts


def _classify(values):
    if all(v == 0 for v in values):
        return 'all_zero'
    if all(v == 1 for v in values):
        return 'all_one'
    return 'other_tie' if max(values) == min(values) else 'non_tie'


def score_audit(predictions, outcomes, rule_index, probabilities=None):
    """Independent candidate ranking; tied outcomes never create preferences."""
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    means = outcomes.mean(axis=1)
    differences = outcomes-outcomes[rule_index]
    total = credit = 0
    for i in range(len(means)):
        for j in range(i):
            if means[i] == means[j]:
                continue
            total += 1
            gap = predictions[i]-predictions[j]
            credit += .5 if abs(gap) <= 1e-12 else float(gap*(means[i]-means[j]) > 0)
    result = dict(non_tie_pairs=total, ranking_credit=credit,
        ranking_accuracy=credit/total if total else None,
        paired_advantage_mse=float(np.mean((predictions-differences.mean(axis=1))**2)),
        probability_brier=None, probability_ece=None)
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)
        if p.shape != means.shape or np.any((p < 0) | (p > 1)):
            raise ValueError('BCE calibration requires one valid probability per candidate')
        bins = []
        for index in range(10):
            mask = (p >= index/10) & ((p < (index+1)/10) if index < 9 else (p <= 1.))
            bins.append(dict(count=int(mask.sum()), prediction_sum=float(p[mask].sum()),
                             outcome_sum=float(means[mask].sum())))
        result.update(probability_brier=float(np.mean((p-means)**2)),
                      probability_ece=sum(abs(b['prediction_sum']-b['outcome_sum']) for b in bins)/len(p),
                      calibration_bins=bins)
    return result


def diagnose_policy(ctx, policy, method_id, candidate_provider=None, include_own=True, **kwargs):
    from .tasks.t1_rollout import load_historical_controls
    frozen, _ = load_historical_controls(ctx.config)
    all_results = {}
    sets = [('diagnostic', ctx.config['diagnostic_states'], 'shared')]
    if include_own:
        sets.append(('onpolicy_diagnostic',ctx.config['own_diagnostic_states'],f'{ctx.task_id}:{method_id}'))
    for split, count, namespace in sets:
        samples = ctx.collect_states(count, split, policy=None if namespace == 'shared' else policy, namespace=namespace)
        result_rows = []
        directory = ctx.output/'diagnostics'/method_id/split
        directory.mkdir(parents=True,exist_ok=True)
        for index, sample in enumerate(samples):
            path = directory/f'{index:05d}.pt'
            if path.exists():
                result_rows.append(torch.load(path,map_location='cpu',weights_only=False)['summary'])
                continue
            env = env_from_snapshot(sample['snapshot'])
            state = env.state()
            selected = choose(policy,env)
            env.close()
            pool = propose_plans(state,8,ctx.seed,frozen.get('frozen_b1'))
            if 'frozen_b3' in frozen:
                old = frozen['frozen_b3'].act(state)
                if old not in pool:
                    pool.append(old)
            base_count = len(pool)
            extras = list(candidate_provider(state)) if candidate_provider else []
            for item in [*extras,selected]:
                if partition_key(item) not in {partition_key(x) for x in pool}:
                    pool.append(item)
            rule_index = next(i for i,p in enumerate(pool) if p == rule_grouping(state))
            choice = pool.index(selected)
            provider_indices = sorted({pool.index(p) for p in extras})
            selection_seeds = [stable_seed(ctx.seed,'diagnostic_selection',sample['state_id'],b)
                               for b in range(ctx.config['selection_branches'])]
            verification_seeds = [stable_seed(ctx.seed,'diagnostic_verification',sample['state_id'],b)
                                  for b in range(ctx.config['verification_branches'])]
            selection = paired_rollouts(sample['snapshot'],pool,branch_seeds=selection_seeds)
            verification = paired_rollouts(sample['snapshot'],pool,branch_seeds=verification_seeds)
            values = [x['y'] for x in selection]
            teacher = choose_with_rule_ties(state,pool,values)
            provider_teacher = (provider_indices[choose_with_rule_ties(state,[pool[i] for i in provider_indices],
                                [values[i] for i in provider_indices])] if provider_indices else None)
            summary = dict(family_id=sample['family_id'],state_id=sample['state_id'],state_kind=sample['state_kind'],
                red_count=sample['episode_spec']['red_count'],blue_count=sample['episode_spec']['blue_count'],
                state_set=split,method_id=method_id,label_class=_classify(values),
                base_candidate_count=base_count,extended_candidate_count=len(pool),
                selection_branches=len(selection_seeds),verification_branches=len(verification_seeds),
                selected_vs_rule_verification_gain=verification[choice]['y']-verification[rule_index]['y'],
                teacher_minus_method_verification_gap=verification[teacher]['y']-verification[choice]['y'],
                verification_rule=verification[rule_index]['y'],verification_selected=verification[choice]['y'],
                teacher_selection_index=teacher,selected_index=choice,
                provider_unique_candidates=len(provider_indices),
                provider_raw_candidates=len(extras),
                provider_teacher_vs_rule_verification_gain=(verification[provider_teacher]['y']-verification[rule_index]['y']
                                                           if provider_teacher is not None else None),
                simulated_physical_steps=sum(sum(r['physical_steps']) for r in selection+verification))
            predictions = probabilities = None
            if hasattr(policy, 'score_candidates'):
                predictions = policy.score_candidates(state,pool)
                if hasattr(policy, 'predict_candidate_probabilities'):
                    probabilities = policy.predict_candidate_probabilities(state,pool)
                summary.update(score_audit(predictions,[r['outcomes'] for r in verification],rule_index,probabilities))
            ctx.checkpoint(str(path.relative_to(ctx.output)),dict(summary=summary,selection=selection,verification=verification,
                predicted_advantages=predictions,predicted_probabilities=probabilities))
            for purpose,rows in [('selection',selection),('verification',verification)]:
                for ci,row in enumerate(rows):
                    ctx.log('candidates',dict(state_id=sample['state_id'],method_id=method_id,
                        candidate_id=ci,purpose=purpose,action=row['action'],success_mean=row['y']))
                    for branch in row['branches']:
                        ctx.log('branches',dict(state_id=sample['state_id'],method_id=method_id,
                            candidate_id=ci,purpose=purpose,**branch))
            result_rows.append(summary)
            ctx.progress('diagnostic',method_id=method_id,diagnostic_split=split,
                         completed_states=len(result_rows),total_states=count)
        summary = dict(method_id=method_id,state_set=split,states=len(result_rows),
            families=len({r['family_id'] for r in result_rows}),
            label_classes={name:sum(r['label_class']==name for r in result_rows)
                           for name in ('all_zero','all_one','other_tie','non_tie')},
            selected_vs_rule_verification_gain=float(np.mean([r['selected_vs_rule_verification_gain'] for r in result_rows])),
            teacher_minus_method_verification_gap=float(np.mean([r['teacher_minus_method_verification_gap'] for r in result_rows])),
            simulated_physical_steps=sum(r['simulated_physical_steps'] for r in result_rows),
            complete=len(result_rows)==count)
        scored = [r for r in result_rows if 'non_tie_pairs' in r]
        if scored:
            pairs = sum(r['non_tie_pairs'] for r in scored)
            summary.update(non_tie_pairs=pairs,
                ranking_accuracy=sum(r['ranking_credit'] for r in scored)/pairs if pairs else None,
                paired_advantage_mse=float(np.mean([r['paired_advantage_mse'] for r in scored])),
                ranking_weighting='non-tie candidate pairs; prediction ties receive half credit')
            calibrated = [r for r in scored if r['probability_brier'] is not None]
            if calibrated:
                bins = [{key:sum(r['calibration_bins'][i][key] for r in calibrated)
                         for key in ('count','prediction_sum','outcome_sum')} for i in range(10)]
                summary.update(probability_brier=float(np.mean([r['probability_brier'] for r in calibrated])),
                    probability_ece=sum(abs(b['prediction_sum']-b['outcome_sum']) for b in bins)/sum(b['count'] for b in bins),
                    calibration_bins=bins,probability_weighting='state-equal Brier; candidate-equal ECE')
        atomic_json(directory/'summary.json',summary)
        all_results[split]=summary
    return all_results
