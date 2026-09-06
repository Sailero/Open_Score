"""First-gate checks: distinguish task allocation from useful partition edits."""
from __future__ import annotations

import time
import numpy as np
from open_score.grouping.domain import Group, Grouping
from open_score.grouping.storage import atomic_json
from .actions import candidate_pool, grand_grouping, rule_grouping
from .config import provenance
from .environment import make_env
from .policies import RulePolicy


def _variant(state, mode):
    base = grand_grouping(state)
    if mode == 'grand':
        return base
    if mode == 'singletons':
        return Grouping(tuple(Group(g.target, (i,)) for g in base.groups for i in g.members))
    if mode == 'pairs':
        # Diagnostic partition with exactly the same target membership, not an
        # action-space constraint or a privileged controller for the algorithm.
        positions = {e.id: np.asarray(e.position) for e in state.alive('red')}
        groups = []
        for group in base.groups:
            remaining = set(group.members)
            while remaining:
                first = min(remaining)
                remaining.remove(first)
                members = [first]
                if remaining:
                    second = min(remaining, key=lambda i: (np.linalg.norm(positions[i]-positions[first]), i))
                    remaining.remove(second)
                    members.append(second)
                groups.append(Group(group.target, tuple(members)))
        return Grouping(tuple(groups))
    return rule_grouping(state)


def run_diagnostics(output, *, episodes=8, scales=(4, 8), seed=20260906,
                    opponent='reactive', branches=2, **kwargs):
    """Actual full episodes and paired same-state full terminal continuations.

    No threshold can prove grouping has independent value from this pilot.
    Reports observed differences and an explicit inconclusive outcome instead.
    """
    from pathlib import Path
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started, rows, counterfactuals = time.monotonic(), [], []
    modes = ('grand', 'singletons', 'pairs', 'rule')
    for scale in scales:
        env = make_env(int(scale), opponent=opponent)
        for index in range(int(episodes)):
            episode_seed = int(seed)+int(scale)*10000+index
            for mode in modes:
                state = env.reset(episode_seed)
                while not env.done:
                    state, reward, _, info = env.step(_variant(state, mode))
                rows.append(dict(scale=int(scale), seed=episode_seed, mode=mode,
                                 success=bool(info['success']), steps=state.step))
            # Compare partitions at SAME physical states, not merely the same
            # initial seed after the policies have already diverged.
            state = env.reset(episode_seed)
            for event in range(3):
                if env.done:
                    break
                snapshot = env.snapshot()
                proposals = {mode: _variant(state, mode) for mode in modes}
                action_rows = {mode: env.executor.act(env.adapter, action) for mode, action in proposals.items()}
                env.restore(snapshot)
                base_actions = action_rows['grand']
                for mode, action in proposals.items():
                    outcomes = []
                    for branch in range(int(branches)):
                        env.restore(snapshot)
                        env.set_rng(episode_seed*101+event*17+branch)
                        following, _, _, info = env.step(action)
                        while not env.done:
                            following, _, _, info = env.step(rule_grouping(following))
                        outcomes.append(int(info['success']))
                    counterfactuals.append(dict(scale=int(scale), family=episode_seed,
                        state_step=state.step, mode=mode, outcomes=outcomes,
                        action_difference=sum(action_rows[mode][i] != base_actions[i] for i in state.ids('red'))/max(1,len(state.ids('red'))),
                        same_target_membership=action.assignment()==proposals['grand'].assignment()))
                env.restore(snapshot)
                state, _, _, _ = env.step(rule_grouping(state))
        env.close()
        atomic_json(output/'partial.json', dict(rows=rows, counterfactuals=counterfactuals))
    summary = []
    for scale in scales:
        for mode in modes:
            subset = [r for r in rows if r['scale']==int(scale) and r['mode']==mode]
            summary.append(dict(scale=int(scale), mode=mode, episodes=len(subset),
                                wins=sum(r['success'] for r in subset),
                                win_rate=float(np.mean([r['success'] for r in subset]))))
    paired = {}
    for row in counterfactuals:
        paired.setdefault((row['scale'],row['family'],row['state_step']),{})[row['mode']] = row
    differing = sum(any(r['outcomes'] != sample['grand']['outcomes'] for mode,r in sample.items()
                        if mode != 'grand' and r['same_target_membership']) for sample in paired.values())
    report = dict(schema='v4-rule-grouping-diagnostic', provenance=provenance(),
                  executor='rule_group_v1', opponent=opponent, max_steps=50,
                  rows=rows, counterfactuals=counterfactuals, summary=summary,
                  paired_states=len(paired), outcome_sensitive_states=differing,
                  conclusion=('Partition-sensitive continuations observed; advantage over grand groups remains to be tested.' if differing else
                              'No partition-sensitive terminal differences observed in this pilot; grouping benefit is unestablished.'),
                  elapsed_seconds=time.monotonic()-started,
                  interpretation='Pilot, not a significance test. Do not claim grouping innovation from differing physical actions alone.')
    atomic_json(output/'diagnostics.json', report)
    lines = ['# Rule lower and grouping diagnostic', '', report['conclusion'], '',
             '| Scale | Partition | Wins | Episodes | Win rate |','|---|---|---|---|---|']
    lines.extend(f"| {r['scale']} | {r['mode']} | {r['wins']} | {r['episodes']} | {r['win_rate']:.1%} |" for r in summary)
    lines.extend(['', f"Same-state terminal differences: {differing}/{len(paired)}.", '', report['interpretation']])
    (output/'diagnostics.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report
