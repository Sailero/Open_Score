"""Pinned provenance of reference mechanisms; no claim of original-environment replication."""
from pathlib import Path
import json
from open_score.research_v4.runner import atomic_json

SOURCES = {
    'rollout': dict(url='https://web.mit.edu/dimitrib/www/Rollout_Short_View.pdf',kind='paper',implementation='paper_algorithm_reimplementation'),
    'rsapi': dict(url='https://arxiv.org/abs/0805.2027',kind='paper',implementation='fixed_budget_adaptation'),
    'dpw': dict(url='https://github.com/JuliaPOMDP/MCTS.jl',commit='530050351e64f6f6e2bb565051759bc44621cdec',
                license='MIT (Expat); upstream LICENSE.md retains conflict-marker author lines',
                implementation='Python algorithm reimplementation; Julia package not installed'),
    'ppo': dict(url='https://arxiv.org/abs/1707.06347',kind='paper',implementation='event_joint_action_PPO_reimplementation'),
    'attention': dict(url='https://github.com/wouterkool/attention-learn-to-route',commit='c9abf41ac2f878a55b20dc7e829bc942bb999631',
                      license='MIT',implementation='construction representation adapted; original REINFORCE optimizer not replicated'),
    'masking': dict(url='https://arxiv.org/abs/2006.14171',kind='paper',implementation='legal conditional action masks'),
    'exit': dict(url='https://arxiv.org/abs/1705.08439',kind='paper',implementation='rollout approximate policy iteration adaptation'),
    'bridge': dict(url='https://proceedings.iclr.cc/paper_files/paper/2026/file/6960aa296f19962a8efcb21445acdee3-Paper-Conference.pdf',
                   kind='paper',implementation='upper merge STOP DQN only; fixed lower and global continuation potential'),
    'spibb': dict(url='https://github.com/RomainLaroche/SPIBB',commit='833f32b642759111907d2c4d086d1ae1ce9d84cb',
                  license='MIT',implementation='baseline selection idea only; not SPIBB and no SPIBB guarantee'),
}
TASK_REFERENCES={'T1':['rollout','rsapi'],'T2':['dpw'],'T3':['ppo','attention','masking'],
                 'T4':['exit','rsapi'],'T5':['bridge'],'T6':['rsapi','spibb']}

MECHANISMS={
 'T1':'Complete terminal rollout after one candidate event, followed by fixed rule continuation; empirical rule-priority ties.',
 'T2':'Action and stochastic-successor double progressive widening, reused sampled successor nodes, terminal rule continuation leaves.',
 'T3':'Independent actor/critic PPO; matched candidate and unrestricted autoregressive grouping; PPO clipping uses joint event likelihood.',
 'T4':'Independent on-policy expert generation, then fixed-epoch distillation; rollout teacher adaptation of expert iteration.',
 'T5':'Same-target merge/STOP DQN with fixed target identities and paired terminal potential differences; lower executor remains rule based.',
 'T6':'Same family-split paired counterfactual data, BCE versus advantage MSE, identical search, rule anchor and validation-only gate selection.'}


def register(ctx,append_markdown=False):
    rows={k:SOURCES[k] for k in TASK_REFERENCES[ctx.task_id]}
    atomic_json(ctx.output/'source_references.json',dict(task_id=ctx.task_id,references=rows,
                original_environments_run=False,code_vendored=False,protocol=ctx.identity))
    path=ctx.output/'reproduction_map.md'
    if not path.exists():
        path.write_text('\n'.join([f'# {ctx.task_id} reproduction and migration map','',
            MECHANISMS[ctx.task_id],'',
            'Scope: known reactive opponent; shared rule executor; identity-level grouping and reserve; 15 mixed cells; native 50-step terminal success.',
            'This is a mechanism reimplementation/transfer. Original benchmark environments are not run and original-paper scores are not claimed.',
            'Parameters below are frozen before outcomes. Differences from source methods are necessary to represent variable groups, physical command events and known-opponent continuation.',
            'Correctness evidence: tests/test_v5_planning.py, test_v5_neural.py, test_v5_learning_tasks.py, test_v5_value_tasks.py and test_v5_runtime.py.',
            '', '```json',json.dumps(ctx.config,ensure_ascii=False,indent=2),'```','']),encoding='utf-8')
        append_markdown=True
    if append_markdown:
        current=path.read_text(encoding='utf-8') if path.exists() else f'# {ctx.task_id} reproduction map\n'
        marker='<!-- PINNED_REFERENCES -->'
        current=current.split(marker)[0].rstrip()
        block=[marker,'','## Pinned reference provenance','',
               'Mechanisms are reimplemented/adapted in this environment. Original paper environments and full original algorithms are not claimed as reproduced.','']
        for key,row in rows.items():
            block.append(f"- [{key}]({row['url']}): {row.get('commit','paper reference')}; {row.get('license','no source code copied')}; {row['implementation']}.")
        path.write_text(current+'\n\n'+'\n'.join(block)+'\n',encoding='utf-8')
