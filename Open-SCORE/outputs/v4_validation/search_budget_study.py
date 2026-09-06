"""Small paired Q^rule search-budget audit; no model fitting or tuning.

Run from any directory with the project's Torch interpreter. Each proposal is
executed for one real command event, followed by the same fixed rule to native
terminal. Identical selected actions share actual paired rollouts, not extra
independent samples. Model scores are never used as outcome labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch

from open_score.grouping.storage import atomic_json, append_jsonl, sha256
from open_score.research_v4.actions import rule_grouping, search, partition_key
from open_score.research_v4.config import provenance
from open_score.research_v4.data import collect_counterfactuals
from open_score.research_v4.environment import make_env
from open_score.research_v4.outcomes import load_evaluator


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def run(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('choose a new empty study directory; results are immutable')
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = {name: Path(args.shared)/f'{kind}_model/best.pt'
                   for name, kind in (('B2_local', 'local'), ('B3_global', 'global'))}
    hashes = {name: sha256(path) for name, path in checkpoints.items()}
    source = provenance()
    scorers = {name: load_evaluator(path, args.device) for name, path in checkpoints.items()}
    environments, cases = [], []
    behavior_steps = 0
    try:
        for scale in args.scales:
            for episode in range(args.episodes):
                # Separate from all training, validation and final-test streams.
                seed = args.seed + scale*1000 + episode
                env = make_env(scale, seed=seed)
                environments.append(env)
                state = env.reset(seed=seed)
                for event in range(args.states):
                    cases.append((event, episode, scale, seed, env, state, env.snapshot()))
                    if event+1 == args.states:
                        break
                    state, _, done, info = env.step(rule_grouping(state))
                    behavior_steps += int(info['delta'])
                    if done:
                        break
        # All scales receive initial-state coverage before adding later states.
        cases.sort(key=lambda case: case[:3])
        if cases:
            warm_state = cases[0][5]
            for scorer in scorers.values():
                scorer(warm_state, [rule_grouping(warm_state)])
            synchronize(args.device)
        results, branch_steps, branch_rollouts, evaluated_states = [], 0, 0, []
        for event, episode, scale, seed, env, state, snapshot in cases:
            if time.monotonic()-started >= args.max_seconds:
                break
            env.restore(snapshot)
            selections = []
            for method, scorer in scorers.items():
                for budget in args.budgets:
                    # Cache may be reused *within* one search, but not gifted
                    # to a larger budget by an earlier independent search.
                    scorer._cache_identity = None
                    synchronize(args.device)
                    before = time.perf_counter()
                    action, trace = search(state, scorer, budget=budget, return_trace=True)
                    synchronize(args.device)
                    selections.append({'method': method, 'budget': budget,
                        'action': action, 'evaluations': int(trace['evaluations']),
                        'decision_ms': 1000.*(time.perf_counter()-before)})
            before = time.perf_counter()
            rule = rule_grouping(state)
            selections.append({'method': 'rule', 'budget': 0, 'action': rule,
                               'evaluations': 0, 'decision_ms': 1000.*(time.perf_counter()-before)})
            unique, indices = [], {}
            for selection in selections:
                key = partition_key(selection['action'])
                if key not in indices:
                    indices[key] = len(unique)
                    unique.append(selection['action'])
            branch_seed = args.seed + 20_000_000 + scale*10_000 + episode*100 + event
            stream = np.random.default_rng(branch_seed)
            branch_seeds = stream.choice(np.iinfo(np.int32).max, size=args.branches, replace=False)
            # This is paired_terminal's underlying API, retained here so actual
            # individual terminal outcomes are auditable instead of just means.
            labeled = collect_counterfactuals(env, state, unique, branch_seeds)
            reference = labeled[indices[partition_key(rule)]]
            branch_steps += sum(sum(row['physical_steps']) for row in labeled)
            branch_rollouts += len(labeled)*args.branches
            state_id = f'{scale}:{episode}:{event}:{state.step}'
            for selection in selections:
                action_index = indices[partition_key(selection['action'])]
                label = labeled[action_index]
                difference = np.asarray(label['outcomes'])-np.asarray(reference['outcomes'])
                results.append({'state_id': state_id, 'scale': scale, 'episode': episode,
                    'episode_seed': seed, 'event_index': event, 'physical_step': state.step,
                    'method': selection['method'], 'search_budget': selection['budget'],
                    'candidate_evaluations': selection['evaluations'],
                    'decision_ms': selection['decision_ms'], 'branches': args.branches,
                    'terminal_success_mean': label['y'], 'rule_terminal_success_mean': reference['y'],
                    'paired_difference_vs_rule': float(difference.mean()),
                    'successes': sum(label['outcomes']), 'rule_successes': sum(reference['outcomes']),
                    'shared_terminal_action_index': action_index,
                    'action_sha256': hashlib.sha256(json.dumps(selection['action'].to_dict(), sort_keys=True).encode()).hexdigest()})
            append_jsonl(output/'paired_rollouts.jsonl', {'state_id': state_id, 'episode_seed': seed,
                         'physical_step': state.step, 'unique_actions': len(unique),
                         'reference_action_index': indices[partition_key(rule)], 'labels': labeled})
            evaluated_states.append(state_id)
            print(f'{state_id}: {len(unique)} unique choices, {len(labeled)*args.branches} terminal rollouts; '
                  f'elapsed {time.monotonic()-started:.1f}s', flush=True)
        if not results:
            raise RuntimeError('no real states evaluated within the requested time budget')
        with (output/'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        aggregates = []
        for method in ('B2_local', 'B3_global'):
            for budget in args.budgets:
                selected = [r for r in results if r['method'] == method and r['search_budget'] == budget]
                aggregates.append({'method': method, 'search_budget': budget, 'states': len(selected),
                    'paired_branch_comparisons': sum(r['branches'] for r in selected),
                    'terminal_success_mean': float(np.mean([r['terminal_success_mean'] for r in selected])),
                    'paired_difference_vs_rule': float(np.mean([r['paired_difference_vs_rule'] for r in selected])),
                    'candidate_evaluations_mean': float(np.mean([r['candidate_evaluations'] for r in selected])),
                    'decision_ms_mean': float(np.mean([r['decision_ms'] for r in selected])),
                    'decision_ms_p95': float(np.percentile([r['decision_ms'] for r in selected], 95))})
        after_hashes = {name: sha256(path) for name, path in checkpoints.items()}
        if after_hashes != hashes or provenance()['source_hash'] != source['source_hash']:
            raise RuntimeError('source or model changed during audit')
        summary = {'schema': 'v4-search-budget-paired-q-rule-v1', 'status': 'complete',
            'settings': vars(args), 'source': source,
            'checkpoints': {name: {'path': str(path), 'sha256': hashes[name]} for name, path in checkpoints.items()},
            'models_unchanged': True, 'planned_states': len(cases), 'evaluated_states': len(evaluated_states),
            'evaluated_state_ids': evaluated_states,
            'episodes_represented': len({(r['scale'], r['episode']) for r in results}),
            'states_by_scale': {str(scale): len({r['state_id'] for r in results if r['scale'] == scale}) for scale in args.scales},
            'aggregates': aggregates,
            'rule_terminal_success_mean': float(np.mean([r['terminal_success_mean'] for r in results if r['method'] == 'rule'])),
            'physical_cost': {'behavior_collection_steps': behavior_steps,
                'counterfactual_steps': branch_steps, 'total_steps': behavior_steps+branch_steps,
                'actual_terminal_rollouts': branch_rollouts, 'selected_policy_comparisons': len(results)*args.branches},
            'elapsed_seconds': time.monotonic()-started,
            'interpretation': 'small-sample complete terminal Q^rule rollout audit: proposal one event, then fixed rule. Not full-policy episode win rate, not converged learning or true oracle.',
            'pairing': 'same state and independent common branch seeds across all selected actions; identical actions reuse same physical outcomes',
            'latency_protocol': 'one warmup per scorer; synchronize CUDA; clear cross-search local cache before each budget; retain within-search cache',
            'statistical_limit': 'two original episodes per scale, at most two correlated states per episode, two branches; descriptive curves only, no significance claim'}
        atomic_json(output/'summary.json', summary)
        plot(summary, output/'budget_curves.png')
        print(json.dumps({'output': str(output), 'elapsed_seconds': summary['elapsed_seconds'],
                          'states': summary['evaluated_states'], 'physical_cost': summary['physical_cost'],
                          'aggregates': aggregates}, ensure_ascii=False), flush=True)
    finally:
        for env in environments:
            env.close()


def plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    definitions = [('terminal_success_mean', 'Actual terminal success under Q^rule rollout', 100.),
                   ('paired_difference_vs_rule', 'Paired terminal difference vs rule (percentage points)', 100.),
                   ('decision_ms_mean', 'Mean decision latency (ms)', 1.),
                   ('candidate_evaluations_mean', 'Actual candidate evaluations', 1.)]
    for ax, (metric, title, multiplier) in zip(axes.flat, definitions):
        for method, label in (('B2_local', 'B2: local outcome product'), ('B3_global', 'B3: global continuation value')):
            rows = [r for r in summary['aggregates'] if r['method'] == method]
            ax.plot([r['search_budget'] for r in rows], [multiplier*r[metric] for r in rows], 'o-', label=label)
        if metric == 'terminal_success_mean':
            ax.axhline(summary['rule_terminal_success_mean']*100., color='gray', linestyle='--', label='Rule continuation reference')
            ax.set_ylabel('Success (%)')
        elif metric == 'paired_difference_vs_rule':
            ax.axhline(0., color='gray', linestyle='--')
        ax.set(xlabel='Search evaluation budget', title=title)
        ax.set_xticks(summary['settings']['budgets'])
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Small-sample search-budget study: actual complete-terminal Q^rule rollouts', fontsize=13)
    fig.text(.5, .015, f"{summary['evaluated_states']} states from {summary['episodes_represented']} original episodes; "
             f"{summary['settings']['branches']} paired branches per action. Correlated states; descriptive evidence only.",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0., .045, 1., .95))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', default=str(ROOT/'outputs/v4_validation/full_scale_pilot/shared/seed_20260906'))
    parser.add_argument('--output', default=str(ROOT/'outputs/v4_validation/search_budget_study'))
    parser.add_argument('--scales', nargs='+', type=int, default=[8, 12, 16, 24, 32])
    parser.add_argument('--budgets', nargs='+', type=int, default=[8, 32, 64])
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--states', type=int, default=2)
    parser.add_argument('--branches', type=int, default=2)
    parser.add_argument('--seed', type=int, default=90260906)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=180.)
    args = parser.parse_args()
    if min(args.episodes, args.states, args.branches, args.threads, *args.scales, *args.budgets) < 1:
        parser.error('counts, scales and budgets must be positive')
    run(args)


if __name__ == '__main__':
    main()
