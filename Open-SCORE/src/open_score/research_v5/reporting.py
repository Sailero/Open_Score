"""Incremental reports: committed observations, explicit axes and evidence limits."""
from __future__ import annotations

from collections import defaultdict, deque
import csv
from datetime import datetime, timezone
import gzip
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from scipy.stats import beta
from open_score.grouping.storage import replace_file
from open_score.research_v4.evaluation import read_records, wilson
from open_score.research_v5.protocol import ROOT

MAIN = {'T1_rollout':'latest', 'T2_mcts_dpw':'latest', 't3_candidate':'latest',
    't3_autoregressive':'latest', 't4_exit':'latest', 't5_bridge_grouping':'final_construction',
    't6_bce':'final_epoch', 't6_adv':'final_epoch', 't6_adv_gated':'final_epoch',
    'rule':'frozen', 'grand':'frozen', 'singleton':'frozen', 'frozen_b1':'frozen', 'frozen_b3':'frozen'}
TASK_METHODS = {'T1':['T1_rollout','rule','grand','singleton','frozen_b1','frozen_b3'],
    'T2':['T2_mcts_dpw'], 'T3':['t3_candidate','t3_autoregressive'], 'T4':['t4_exit'],
    'T5':['t5_bridge_grouping'], 'T6':['t6_bce','t6_adv','t6_adv_gated']}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def records(path):
    try:
        return read_records(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_bytes(content.encode('utf-8'))
    replace_file(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows, fields=None):
    fields = fields or sorted({k for row in rows for k in row}) or ['status']
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (dict,list,tuple))
                         else row.get(k) for k in fields})
    atomic_text(path, '\ufeff'+text.getvalue())


def finite(value):
    return isinstance(value, (int,float,np.integer,np.floating)) and math.isfinite(float(value))


def mean(values):
    values = [float(v) for v in values if finite(v)]
    return float(np.mean(values)) if values else None


def success(row):
    return bool(row.get('success_native', row.get('success', False)))


def signature(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def clopper_pearson(k, n, alpha=.025):
    if not n:
        return [None, None]
    return [0. if k == 0 else float(beta.ppf(alpha/2,k,n-k+1)),
            1. if k == n else float(beta.ppf(1-alpha/2,k+1,n-k))]


def paired_interval(method, baseline):
    lookup = {r['family_id']:r for r in baseline}
    pairs = [(success(r), success(lookup[r['family_id']])) for r in method if r['family_id'] in lookup]
    counts = {f'n{a}{b}':sum(x==bool(a) and y==bool(b) for x,y in pairs) for a in (0,1) for b in (0,1)}
    n = len(pairs)
    if not n:
        return dict(pairs=0, **counts, difference=None, interval95=[None,None])
    pos, neg = clopper_pearson(counts['n10'],n), clopper_pearson(counts['n01'],n)
    return dict(pairs=n, **counts, difference=(counts['n10']-counts['n01'])/n,
        interval95=[pos[0]-neg[1],pos[1]-neg[0]],
        interval_method='Bonferroni combination of 97.5% Clopper-Pearson bounds')


def episode_rows(directory):
    """Family shards commit before the append-only episode index."""
    rows = {r['family_id']:r for r in records(directory/'episodes.jsonl') if 'family_id' in r}
    indexed = {str(r.get('raw_shard','')).replace('\\','/') for r in rows.values()}
    for shard in (directory/'families').glob('*.jsonl.gz'):
        relative = shard.relative_to(directory).as_posix()
        if relative in indexed:
            continue
        try:
            with gzip.open(shard,'rt',encoding='utf-8') as stream:
                row = json.loads(stream.readline())
            row['raw_shard'] = relative
            rows[row['family_id']] = row
        except (OSError,EOFError,KeyError,json.JSONDecodeError):
            continue
    return list(rows.values())


def latency(directory, rows, cache):
    """Exact event quantiles, cached by immutable shard; no median-of-medians."""
    values, families = [], {r['family_id'] for r in rows}
    shards = list((directory/'families').glob('*.jsonl.gz'))
    if shards:
        for path in shards:
            key, sig = str(path.resolve()), signature(path)
            saved = cache.get(key)
            if not saved or saved['signature'] != sig:
                times, family = [], None
                try:
                    with gzip.open(path,'rt',encoding='utf-8') as stream:
                        for line in stream:
                            row = json.loads(line)
                            if row.get('record_type') == 'episode':
                                family = row.get('family_id')
                            if row.get('record_type') == 'event' and finite(row.get('selection_wall_ms')):
                                times.append(float(row['selection_wall_ms']))
                except (OSError,EOFError,json.JSONDecodeError):
                    continue
                saved = dict(signature=sig, family_id=family, values=times)
                cache[key] = saved
            if saved['family_id'] in families:
                values.extend(saved['values'])
    elif (directory/'events.jsonl').exists():
        path = directory/'events.jsonl'
        key, sig = str(path.resolve()), signature(path)
        saved = cache.get(key)
        if not saved or saved['signature'] != sig:
            unique = {r.get('event_id',f'row_{i}'):r for i,r in enumerate(records(path))}
            saved = dict(signature=sig, values=[float(r['selection_wall_ms']) for r in unique.values()
                                               if finite(r.get('selection_wall_ms'))])
            cache[key] = saved
        values.extend(saved['values'])
    return dict(decision_samples=len(values), decision_p50_ms=float(np.quantile(values,.5)) if values else None,
        decision_p95_ms=float(np.quantile(values,.95)) if values else None,
        decision_latency_source='recorded_command_events' if values else 'not_recorded',
        episode_median_latency_mean_ms=mean(r.get('decision_time_p50_ms') for r in rows))


def effect(paired, complete, reference_complete, smoke, reference=False):
    if smoke:
        return 'smoke_only'
    if reference:
        return 'reference'
    if not complete or not reference_complete or not paired['pairs']:
        return 'inconclusive'
    if paired['interval95'][0] > 0:
        return 'observed_gain_single_seed'
    return 'no_observed_gain' if paired['difference'] <= 0 else 'inconclusive'


def task_status(path):
    progress, result, summary = (read_json(path/name) for name in ('progress.json','task_result.json','summary.json'))
    if summary.get('generated_by')=='incremental_reporting': summary={}
    if progress.get('execution_status') == 'failed_execution' or progress.get('phase') == 'execution_error':
        state = 'failed_execution'
    elif result.get('complete') or summary.get('status') == 'complete' or summary.get('execution_status') == 'completed' or progress.get('phase') in ('complete','completed'):
        state = 'complete'
    else:
        state = 'running' if progress else 'not_started'
    return dict(task_id=path.name, execution_status=state, phase=progress.get('phase','not_started'),
        updated=progress.get('updated'), finished=result.get('finished'),
        error=progress.get('error') if state == 'failed_execution' else None, result=result.get('result',summary))


def comparison_rows(found, config, cache):
    cells = [tuple(c) for c in config.get('cells',[])]
    per_cell = int(config.get('eval_per_cell',100))
    expected = per_cell*len(cells)
    baseline = found.get(('rule','frozen'),{}).get('rows',[])
    baseline_complete = len(baseline) == expected and expected > 0 and all(
        sum((r['red_count'],r['blue_count'])==cell for r in baseline)==per_cell for cell in cells)
    output = []
    for (method,checkpoint),item in sorted(found.items()):
        rows, cell_rows = item['rows'], []
        wins = sum(map(success,rows))
        for red,blue in cells:
            subset = [r for r in rows if (r['red_count'],r['blue_count']) == (red,blue)]
            ref = [r for r in baseline if (r['red_count'],r['blue_count']) == (red,blue)]
            k, n = sum(map(success,subset)), len(subset)
            cell_rows.append(dict(red_count=red,blue_count=blue,blue_red_ratio=blue/red,
                wins=k,episodes=n,success_rate=k/n if n else None,wilson95=wilson(k,n),
                complete=n==per_cell,paired_vs_rule=paired_interval(subset,ref)))
        complete = bool(expected and len(rows)==expected and all(c['complete'] for c in cell_rows))
        paired, difficulties = paired_interval(rows,baseline), []
        for ratio in sorted({b/r for r,b in cells}):
            selected = [c for c in cell_rows if c['blue_red_ratio']==ratio]
            subset = [r for r in rows if r['blue_count']/r['red_count']==ratio]
            ref = [r for r in baseline if r['blue_count']/r['red_count']==ratio]
            difficulties.append(dict(blue_red_ratio=ratio,cells=len(selected),
                observed_cells=sum(c['episodes']>0 for c in selected),
                macro_success_rate=mean(c['success_rate'] for c in selected),
                wins=sum(map(success,subset)),episodes=len(subset),complete=all(c['complete'] for c in selected),
                paired_vs_rule=paired_interval(subset,ref)))
        equal = next((d for d in difficulties if d['blue_red_ratio']==1.),None)
        output.append(dict(task_id=item['task_id'],method_id=method,checkpoint_id=checkpoint,
            training_seed=config.get('seed'),wins=wins,episodes=len(rows),expected_episodes=expected,
            success_rate=wins/len(rows) if rows else None,wilson95=wilson(wins,len(rows)),
            macro_success_rate=mean(c['success_rate'] for c in cell_rows),
            observed_cells=sum(c['episodes']>0 for c in cell_rows),expected_cells=len(cells),
            equal_count_success_rate=equal['macro_success_rate'] if equal else None,
            equal_count_episodes=equal['episodes'] if equal else 0,
            never_deployed_rate=mean(r.get('never_deployed') for r in rows),
            reserve_exposure_fraction=mean(r.get('reserve_exposure_fraction') for r in rows),
            real_environment_steps=sum(r.get('real_environment_steps',r.get('physical_steps',0)) for r in rows),
            planner_sim_steps=sum(r.get('planner_sim_steps',0) for r in rows),complete=complete,
            paired_vs_rule=paired,evidence_status=effect(paired,complete,baseline_complete and paired['pairs']==len(rows),config.get('smoke',False),method=='rule'),
            cells=cell_rows,difficulties=difficulties,path=str(item['directory']),**latency(item['directory'],rows,cache)))
    return output


def curve_series(task):
    points = {}
    def add(method,phase,unit,x,metric,value,source):
        if finite(x) and finite(value):
            points.setdefault((method,phase,unit,metric),{})[float(x)] = (float(value),source)
    for source in sorted(task.rglob('training*.jsonl')):
        for row in records(source):
            method = row.get('method_id') or ('t6_'+row['kind'] if task.name=='T6' and row.get('kind') else
                                             't5_bridge_grouping' if task.name=='T5' else task.name)
            phase = row.get('phase','supervised' if task.name=='T6' else 'construction' if task.name=='T5' else 'train')
            if phase=='distill':
                phase=f'distill_round_{row.get("round",1)}'
            if 'physical_steps' in row:
                unit,x='Physical training steps',row['physical_steps']
            elif 'construction_episode' in row:
                unit,x='Construction episodes',row['construction_episode']
            elif 'epoch' in row:
                unit,x='Epoch',row['epoch']
            else:
                continue
            for metric in ('loss','actor_loss','value_loss','td_loss','td_error','entropy','joint_entropy',
                           'approx_kl','clip_fraction','gradient_norm','success_window','epsilon','return','final_potential'):
                add(method,phase,unit,x,metric,row.get(metric),source.relative_to(task).as_posix())
            for metric,value in row.get('validation',{}).items():
                if metric in ('brier','ece','advantage_mse','ranking_accuracy','empirical_candidate_regret'):
                    add(method,'validation',unit,x,metric,value,source.relative_to(task).as_posix())
    for row in records(task/'validation.jsonl'):
        unit,x=('Construction episodes',row.get('construction_episode')) if 'construction_episode' in row else ('Physical training steps',row.get('physical_steps'))
        method=row.get('method_id','t5_bridge_grouping' if task.name=='T5' else task.name)
        add(method,'validation',unit,x,
            'native_success_rate',row.get('success_rate'),'validation.jsonl')
    for source in task.glob('evaluations/*/validation/*/summary.json'):
        row=read_json(source)
        checkpoint=row.get('checkpoint_id',source.parent.name)
        match,iteration,construction=re.fullmatch(r'step_(\d+)',checkpoint),re.fullmatch(r'round_(\d+)',checkpoint),re.fullmatch(r'construction_(\d+)',checkpoint)
        if match or iteration or construction:
            unit,x=('Physical training steps',int(match[1])) if match else ('ExIt rounds',int(iteration[1])) if iteration else ('Construction episodes',int(construction[1]))
            add(row.get('method_id',source.parents[2].name),'validation',unit,x,'native_success_rate',
                row.get('success_rate'),source.relative_to(task).as_posix())
    windows=defaultdict(lambda:deque(maxlen=100))
    for row in records(task/'episodes.jsonl'):
        if row.get('phase')=='train' and 'total_training_physical_steps' in row:
            method,cell=row.get('method_id',task.name),f'{row.get("red_count")}v{row.get("blue_count")}'
            windows[method,cell].append(int(success(row)))
            add(method+' '+cell,'train_by_cell','Physical training steps',row['total_training_physical_steps'],
                'rolling_native_success_100',float(np.mean(windows[method,cell])),'episodes.jsonl')
    return [dict(method_id=m,phase=p,x_axis=u,metric=k,points=[[x,y] for x,(y,_) in sorted(v.items())],
                 sources=sorted({s for _,s in v.values()})) for (m,p,u,k),v in sorted(points.items())]


def panel(metric):
    if metric in ('native_success_rate','success_window'): return 'Native task success'
    if metric in ('loss','actor_loss','value_loss','td_loss'): return 'Training losses'
    if metric in ('brier','ece','advantage_mse'): return 'Validation prediction error'
    if metric in ('ranking_accuracy','empirical_candidate_regret'): return 'Validation candidate choice'
    if metric in ('return','final_potential'): return 'Construction continuation value'
    if metric in ('entropy','joint_entropy','epsilon'): return 'Exploration'
    return 'Optimization diagnostics'


def save_figure(fig,path):
    temporary=Path(path).with_name(Path(path).name+f'.{os.getpid()}.tmp')
    fig.savefig(temporary,format='png',dpi=135)
    replace_file(temporary,path)


def budget_rows(task):
    rows=[]
    for r in records(task/'budget_curve.jsonl'):
        if task.name=='T1':
            rows.append(dict(task_id='T1',method_id='T1_rollout',state_id=r.get('state_id'),family_id=r.get('family_id'),
                depth=1,budget_kind='terminal_branches',budget=r.get('branches'),
                verified_gain_vs_rule=r.get('verification_gain_vs_rule'),
                simulation_physical_steps=r.get('selection_simulation_physical_steps'),decision_seconds=None))
        elif task.name=='T2':
            rows.append(dict(task_id='T2',method_id='T2_mcts_dpw',state_id=r.get('state_id'),family_id=r.get('family_id'),
                depth=r.get('depth'),budget_kind='MCTS_iterations',budget=r.get('iterations'),
                verified_gain_vs_rule=r.get('paired_gain_vs_rule'),simulation_physical_steps=r.get('simulation_physical_steps'),
                decision_seconds=r.get('decision_seconds'),iterations_executed=r.get('iterations_executed')))
    return list({(r['state_id'],r['depth'],str(r['budget'])):r for r in rows}.values())


def plot_budgets(task):
    rows=budget_rows(task)
    if not rows:
        return None
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for r in rows: groups[r['depth'],str(r['budget'])].append(r)
    aggregate=[dict(depth=d,budget=rs[0]['budget'],states=len(rs),
        verified_gain_vs_rule=mean(r['verified_gain_vs_rule'] for r in rs),
        mean_simulation_physical_steps=mean(r['simulation_physical_steps'] for r in rs)) for (d,_),rs in sorted(groups.items())]
    atomic_json(task/'budget_curves.json',dict(rows=rows,aggregate=aggregate))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for depth in sorted({r['depth'] for r in aggregate}):
        subset=[r for r in aggregate if r['depth']==depth]
        for axis,key,label in ((axes[0],'budget','Requested search budget'),
                                (axes[1],'mean_simulation_physical_steps','Mean simulated physical steps per decision')):
            values=sorted((float(r[key]),r['verified_gain_vs_rule']) for r in subset if finite(r[key]) and finite(r['verified_gain_vs_rule']))
            if values: axis.plot([x for x,_ in values],[y for _,y in values],'-o',label=f'depth {depth}')
            axis.set(xlabel=label,ylabel='Verified success gain over rule')
            axis.axhline(0,color='gray',linewidth=.7);axis.grid(alpha=.2)
            if values: axis.legend()
    fig.tight_layout();path=task/'budget_curves.png';save_figure(fig,path);plt.close(fig)
    return str(path)


def plot_task(path):
    task=Path(path)
    sources=sorted(set(task.rglob('training*.jsonl'))|set(task.glob('validation.jsonl'))|
        set(task.glob('episodes.jsonl'))|set(task.glob('budget_curve.jsonl'))|
        set(task.glob('evaluations/*/validation/*/summary.json')))
    identity={str(p.relative_to(task)):signature(p) for p in sources}
    output=task/'training_curves.png'
    if read_json(task/'curve_sources.json')==identity and (output.exists() or (task/'budget_curves.png').exists()):
        return str(output if output.exists() else task/'budget_curves.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series=curve_series(task)
    atomic_json(task/'training_curves.json',dict(series=series,notes=[
        'Axes are physical steps, construction episodes, epochs, or ExIt rounds, never a shared row index.',
        'Training, validation, final test, and teacher continuation values are distinct.']))
    panels=defaultdict(list);by_cell=[]
    for s in series:
        if s['phase']=='train_by_cell': by_cell.append(s);continue
        stage='BC' if s['phase']=='rule_bc' else s['phase'] if s['phase'].startswith('distill_round') else ''
        panels[s['x_axis'],panel(s['metric']),stage].append(s)
    if panels:
        height=math.ceil(len(panels)/2)
        fig,axes=plt.subplots(height,2,figsize=(14,max(4,3.8*height)),squeeze=False)
        for axis,((unit,title,stage),items) in zip(axes.flat,sorted(panels.items())):
            for s in items:
                axis.plot([x for x,_ in s['points']],[y for _,y in s['points']],linewidth=1,
                    marker='.' if len(s['points'])<15 else None,label=f'{s["method_id"]}: {s["metric"]}')
            axis.set(title=title+(' / '+stage if stage else ''),xlabel=unit);axis.grid(alpha=.2);axis.legend(fontsize=7)
        for axis in list(axes.flat)[len(panels):]: axis.set_visible(False)
        fig.tight_layout();save_figure(fig,output);plt.close(fig)
    if by_cell:
        methods=sorted({s['method_id'].rsplit(' ',1)[0] for s in by_cell})
        fig,axes=plt.subplots(len(methods),3,figsize=(16,3.5*len(methods)),squeeze=False)
        for i,method in enumerate(methods):
            for j,ratio in enumerate((.5,.75,1.)):
                axis=axes[i,j]
                for s in by_cell:
                    name,cell=s['method_id'].rsplit(' ',1)
                    red,blue=map(int,cell.split('v'))
                    if name==method and blue/red==ratio: axis.plot([x for x,_ in s['points']],[y for _,y in s['points']],label=cell)
                axis.set(title=f'{method}: Blue / Red = {ratio:g}',xlabel='Physical training steps',ylabel='Rolling native success (100 per cell)',ylim=(0,1))
                axis.grid(alpha=.2)
                if axis.lines: axis.legend(fontsize=7)
        fig.tight_layout();save_figure(fig,task/'success_by_difficulty.png');plt.close(fig)
    budget=plot_budgets(task)
    atomic_json(task/'curve_sources.json',identity)
    return str(output) if panels else budget


def diagnostic_rows(run_dir,cache):
    rows=[]
    for path in sorted(run_dir.glob('T*/diagnostics/*/*/*.pt')):
        key,sig=str(path.resolve()),signature(path)
        saved=cache.get(key)
        if not saved or saved['signature']!=sig:
            import torch
            saved=dict(signature=sig,summary=torch.load(path,map_location='cpu',weights_only=False).get('summary',{}))
            cache[key]=saved
        if saved['summary']: rows.append(dict(saved['summary'],task_id=path.relative_to(run_dir).parts[0],source=str(path.relative_to(run_dir))))
    for task in ('T1','T2'):
        for path in sorted((run_dir/task/'own_diagnostics').glob('*.json')):
            r=read_json(path)
            if r: rows.append({**r.get('episode_spec',{}),'task_id':task,'method_id':r.get('method',task),
                'state_set':'onpolicy_diagnostic','state_id':r.get('state_id'),
                'family_id':r.get('family_id',r.get('episode_spec',{}).get('family_id')),
                'selected_vs_rule_verification_gain':r.get('paired_gain_vs_rule'),
                'simulated_physical_steps':r.get('simulation_physical_steps')})
    for path in sorted((run_dir/'T1/common_diagnostics').glob('*.json')):
        r=read_json(path)
        for name,gain in r.get('paired_gain_vs_rule',{}).items():
            rows.append({**r.get('episode_spec',{}),'task_id':'T1','method_id':name,'state_set':'diagnostic',
                'state_id':r.get('state_id'),'family_id':r.get('family_id',r.get('episode_spec',{}).get('family_id')),
                'selected_vs_rule_verification_gain':gain,'label_class':'all_zero' if r.get('selection_all_zero') else
                'other_tie' if r.get('selection_all_tie') else 'non_tie'})
        for name,g in r.get('group_only',{}).items():
            rows.append({**r.get('episode_spec',{}),'task_id':'T1','method_id':name,'state_set':'same_assignment_grouping',
                'state_id':r.get('state_id'),'family_id':r.get('family_id',r.get('episode_spec',{}).get('family_id')),
                'grouping_vs_grand_gain':g.get('mean_difference'),'branch_vector_changed':g.get('branch_vector_changed')})
    for path in sorted((run_dir/'T4').glob('verification_round_*.pt')):
        import torch
        r=torch.load(path,map_location='cpu',weights_only=False)
        rows.append(dict(task_id='T4',method_id='t4_teacher',state_set='teacher_verification',state_id=r.get('state_id'),
            family_id=r.get('family_id'),round=r.get('round'),selected_vs_continuation_verification_gain=r.get('verified_gain'),
            simulated_physical_steps=r.get('physical_steps_simulated')))
    return rows


def reproduction_rows(run_dir,tasks,methods):
    root_files=[p for base in (run_dir,run_dir/'shared') for p in base.glob('*reproduction*')
                if p.is_file() and p.name!='reproduction_status.csv']
    rows=[]
    for task in tasks:
        task_id=task['task_id'];mapping={}
        files=[*root_files,*[p for p in (run_dir/task_id).glob('*reproduction*') if p.is_file()]]
        for path in files:
            if path.suffix=='.json':
                data=read_json(path)
                if isinstance(data,dict):
                    entry=data.get(task_id,data.get('tasks',{}).get(task_id,{}) if isinstance(data.get('tasks',{}),dict) else {})
                    if isinstance(entry,dict): mapping.update(entry)
                    if data.get('task_id',data.get('task'))==task_id: mapping.update(data)
        pinned_path=run_dir/task_id/'source_references.json'
        pinned=read_json(pinned_path)
        if pinned:
            files.append(pinned_path)
            mapping['sources']=pinned.get('references')
        summary=task['result'] if isinstance(task['result'],dict) else {}
        selected=[m for m in methods if m['method_id'] in TASK_METHODS[task_id]]
        rows.append(dict(task_id=task_id,execution_status=task['execution_status'],phase=task['phase'],
            adaptation_scope=mapping.get('adaptation_scope',summary.get('adaptation_scope',summary.get('adaptation'))),
            reproduction_claim=mapping.get('reproduction',summary.get('reproduction')),
            source_reference=mapping.get('reference',mapping.get('sources',summary.get('reference'))),
            mapping_files=[str(p.relative_to(run_dir)) for p in files],source_mapping_recorded=bool(mapping or summary.get('reference')),
            original_environments_run=pinned.get('original_environments_run'),code_vendored=pinned.get('code_vendored'),
            completed_main_arms=sum(m['complete'] for m in selected),expected_main_and_control_arms=len(TASK_METHODS[task_id]),
            evidence_statuses={m['method_id']:m['evidence_status'] for m in selected},
            interpretation='Adaptation / reimplementation is distinct from original-author experiment reproduction.'))
    return rows


def cost_rows(run_dir,tasks):
    output=[]
    for task in tasks:
        name=task['task_id'];payload=task['result'] if isinstance(task['result'],dict) else {}
        progress=read_json(run_dir/name/'progress.json');training=records(run_dir/name/'training.jsonl')
        real_train=0 if name in ('T1','T2','T5','T6') else None
        if name=='T3':
            real_train=sum(max([r.get('physical_steps',0) for r in training if r.get('method_id')==method]+[0]) for method in TASK_METHODS[name])
            if payload.get('arms'): real_train=sum(r.get('physical_steps',0) for r in payload['arms'])
        collection=payload.get('collection_physical_steps',payload.get('state_collection_physical_steps'))
        offline=payload.get('offline_sim_steps',progress.get('offline_sim_steps'))
        if name=='T4':
            collection=payload.get('collection_physical_steps',progress.get('collection_physical_steps'))
            offline=payload.get('teacher_physical_steps',progress.get('teacher_physical_steps'))
        elif name=='T6':
            manifests=[read_json(run_dir/name/'data'/split/'manifest.json') for split in ('train','validation')]
            if any(manifests):
                collection=sum(m.get('state_collection_physical_steps',0) for m in manifests)
                offline=sum(m.get('label_sim_steps',0) for m in manifests)
        combined_offline=offline if name=='T5' else None
        if name=='T5': offline=offline-collection if offline is not None and collection is not None else None
        evaluations=[]
        for path in (run_dir/name).glob('evaluations/*/*/*/episodes.jsonl'): evaluations.extend(episode_rows(path.parent))
        output.append(dict(task_id=name,training_real_physical_steps=real_train,offline_state_collection_physical_steps=collection,
            offline_label_rollout_physical_steps=offline,teacher_verification_physical_steps=payload.get('verification_physical_steps',progress.get('verification_physical_steps')),
            recorded_collection_plus_label_physical_steps=combined_offline,
            evaluation_real_physical_steps=sum(r.get('real_environment_steps',r.get('physical_steps',0)) for r in evaluations),
            evaluation_planner_physical_steps=sum(r.get('planner_sim_steps',0) for r in evaluations),evaluation_episodes=len(evaluations),
            execution_status=task['execution_status'],note='Recorded counters; evaluation includes validation/control. Missing costs are not zero.'))
    return output


def serial_latency(run_dir,methods):
    """Post-worker deployment measurement stays separate from observed event times."""
    source=run_dir/'reports/serial_latency.json'
    measurement=read_json(source)
    entries={r['method_id']:r for r in measurement.get('methods',[]) if 'method_id' in r}
    for method in methods:
        row=entries.get(method['method_id'],{})
        method.update(serial_decision_mean_ms=float(row['mean_seconds'])*1000 if finite(row.get('mean_seconds')) else None,
            serial_decision_p50_ms=float(row['p50_seconds'])*1000 if finite(row.get('p50_seconds')) else None,
            serial_decision_p95_ms=float(row['p95_seconds'])*1000 if finite(row.get('p95_seconds')) else None,
            serial_decision_samples=len(row.get('rows',[])),serial_latency_complete=bool(row.get('complete',False)))
    return dict(complete=bool(measurement.get('complete')) and all(entries.get(m,{}).get('complete',False) for m in MAIN),
        path=source.relative_to(run_dir).as_posix() if source.exists() else None,
        protocol=measurement.get('protocol'),resources_before=measurement.get('resources_before'),
        resources_after=measurement.get('resources_after'),methods=measurement.get('methods',[]))


def task_analyses(run_dir,summary,diagnostics):
    """Evidence-led route notes; generated files cannot mark a worker complete."""
    unresolved={
        'T1':'有限候选覆盖、有限分支误差与持续重规划效应仍需结合预算和单次/持续对照区分。',
        'T2':'搜索深度收益与真实模拟工作量不同；必须依据实际模拟步对照区分深度和预算作用。',
        'T3':'直接策略优化、候选覆盖和状态访问分布的影响，不能只凭 PPO 损失或两臂最终胜率区分。',
        'T4':'教师质量、蒸馏误差与学生状态分布变化仍需逐轮教师/学生配对验证区分。',
        'T5':'只在规则确定的目标和身份归属内学分区；不能据此判断目标分配学习是否有效。',
        'T6':'价值预测/排序误差、候选覆盖和门控作用仍需结合独立 verification 及三臂配对结果区分。'}
    rows=[];full=['# 六路线阶段分析','','以下是实际记录的描述性分析。完成运行不等于有效；仅一个训练种子，不能归因为算法的稳定优势。','']
    for task in summary['tasks']:
        name=task['task_id'];directory=run_dir/name
        if not directory.exists(): continue
        arms=[r for r in summary['methods'] if r['method_id'] in TASK_METHODS[name]]
        relevant=[r for r in diagnostics if r.get('task_id')==name]
        lines=[f"## {name}",'',f"执行：{task['execution_status']}；当前阶段：`{task['phase']}`。",'']
        observations=[];excluded=[]
        for arm in arms:
            p=arm['paired_vs_rule'];lo,hi=p['interval95']
            paired=(f"规则配对 {p['pairs']} 局、差 {p['difference']*100:+.2f} 个百分点，区间 [{lo*100:+.2f}, {hi*100:+.2f}]" if p['pairs'] else '尚无规则配对证据')
            observations.append(f"{arm['method_id']}：原生成功 {arm['wins']}/{arm['episodes']}，{paired}；{arm['evidence_status']}。")
            if arm.get('never_deployed_rate')==0 and arm['episodes']:
                excluded.append(f"{arm['method_id']} 的 {arm['episodes']} 个已观察测试回合均发生部署，已排除这些回合全程后备的退化；不代表已排除无效动作或其他执行问题。")
        groups=defaultdict(list)
        for r in relevant: groups[r.get('method_id',name),r.get('state_set','unspecified')].append(r)
        audits=[]
        for (method,state_set),rs in sorted(groups.items()):
            gap=mean(r.get('teacher_minus_method_verification_gap') for r in rs)
            gain=mean(r.get('selected_vs_rule_verification_gain') for r in rs)
            pairs=sum(r.get('non_tie_pairs',0) for r in rs)
            credit=sum(float(r['ranking_credit']) if finite(r.get('ranking_credit')) else
                float(r['ranking_accuracy'])*r.get('non_tie_pairs',0) if finite(r.get('ranking_accuracy')) else 0 for r in rs)
            audit=dict(method_id=method,state_set=state_set,states=len(rs),teacher_minus_method_gap=gap,
                selected_vs_rule_gain=gain,non_tie_pairs=pairs,ranking_accuracy=credit/pairs if pairs else None,
                all_zero_states=sum(r.get('label_class')=='all_zero' for r in rs),
                non_tie_states=sum(r.get('label_class')=='non_tie' for r in rs))
            audits.append(audit)
            details=[f"{method}/{state_set}：{len(rs)} 条诊断状态记录"]
            if gain is not None: details.append(f"独立复核相对规则平均差 {gain:+.4f}")
            if gap is not None: details.append(f"教师减方法平均差 {gap:+.4f}")
            if pairs: details.append(f"非平局候选对 {pairs}、排序准确率 {credit/pairs:.3f}")
            if any('label_class' in r for r in rs): details.append(f"全零 {audit['all_zero_states']}、非平局 {audit['non_tie_states']}")
            observations.append('；'.join(details)+'。')
        phase_times=defaultdict(float)
        for r in records(directory/'phase_costs.jsonl'):
            wall=r.get('wall_s',r.get('wall_seconds'))
            if finite(wall): phase_times[r.get('phase','unspecified')]+=float(wall)
        dominant=max(phase_times,key=phase_times.get) if phase_times else None
        bottleneck=(f"累计已记录阶段耗时最多为 {dominant}（{phase_times[dominant]:.1f} 秒）；阶段可能嵌套，不将其总和视为总运行时长。" if dominant else '尚无可比较的阶段耗时记录。')
        positive=[r for r in audits if r['teacher_minus_method_gap'] is not None and r['teacher_minus_method_gap']>0]
        if positive: bottleneck+=' 独立复核中存在正教师差，可优先检查选择或拟合误差；有限状态均值不证明原因，也不保证在线可实现该收益。'
        elif audits: bottleneck+=' 当前诊断没有显示正平均教师差的记录，不足以排除未搜索到更优方案或标签噪声。'
        lines.extend(['### 已观察','','\n'.join('- '+s for s in observations) if observations else '尚无末期测试或诊断记录。','',
            '### 已排除的解释','','\n'.join('- '+s for s in excluded) if excluded else '当前记录不足以排除具体失效解释。','',
            '### 当前瓶颈线索与仍无法区分','','瓶颈线索：'+bottleneck,'',unresolved[name],
            '共享规则底层是固定实验条件；不能仅据此排除底层与上层方案交互造成的影响。所有差值仅针对已记录样本及当前训练种子。','',
            f"[任务记录]({name}/)；[诊断明细](diagnostics.csv)；[阶段成本](phase_costs.csv)。",''])
        text='\n'.join(lines);full.append(text)
        row=dict(task_id=name,execution_status=task['execution_status'],observations=observations,
            excluded=excluded,bottleneck_lead=bottleneck,unresolved=unresolved[name],diagnostics=audits)
        rows.append(row)
        if name in ('T3','T4'):
            path=directory/'summary.json';existing=read_json(path)
            if not path.exists() or existing.get('generated_by')=='incremental_reporting':
                atomic_json(path,dict(generated_by='incremental_reporting',updated=summary['updated'],
                    task=name,status=task['execution_status'],phase=task['phase'],worker_result=task['result'],
                    methods=arms,analysis=row,single_training_seed=True))
            analysis_path=directory/'analysis.md';marker='<!-- GENERATED_V5_REPORT -->'
            if not analysis_path.exists() or analysis_path.read_text(encoding='utf-8').startswith(marker):
                local=text.replace(f']({name}/)','](./)').replace('](diagnostics.csv)','](../diagnostics.csv)').replace('](phase_costs.csv)','](../phase_costs.csv)')
                atomic_text(analysis_path,marker+'\n\n'+local)
    atomic_text(run_dir/'task_analysis.md','\n'.join(full))
    return rows


def pct(value): return '未记录' if value is None else f'{value:.2%}'
def number(value): return '未记录' if value is None else f'{value:.3f}'


def markdown(summary):
    methods=summary['methods']
    lines=['# v5.1 实验进度与结果','',f"更新时间：{summary['updated']}。",
        f"执行状态：{'全部完成' if summary['complete'] else '尚未全部完成'}；14 个预定主方法与对照臂，完整测试完成 {sum(r['complete'] for r in methods)} 臂。",'',
        '**仅一个训练种子。执行完成和方法有效分别判定；训练、验证、最终测试分开报告。**',
        '主表使用预先指定的末期模型或冻结策略，没有依据测试结果选择检查点。不同配比总胜率不能与旧 v4 等人数总胜率直接比较。','']
    if summary['smoke']: lines.extend(['**本目录是短测/集成验证，不能作为正式有效性证据。**',''])
    lines.extend(['| 方法 | 检查点 | 成功/局数 | 样本成功率 | 规模配比宏平均 | 等人数宏平均 | 全程未部署 | 测试完整 | 有效性状态 |',
                  '|---|---|---:|---:|---:|---:|---:|---|---|'])
    for r in methods:
        lines.append(f"| {r['method_id']} | {r['checkpoint_id']} | {r['wins']}/{r['episodes']} | {pct(r['success_rate'])} | {pct(r['macro_success_rate'])} | {pct(r['equal_count_success_rate'])} | {pct(r['never_deployed_rate'])} | {r['complete']} | {r['evidence_status']} |")
    if summary['missing_methods']: lines.extend(['','尚无预定末期测试记录：'+', '.join(summary['missing_methods'])+'。'])
    lines.extend(['','未覆盖全部格子时，宏平均仅针对已有记录，不能当作完整矩阵成绩。','',
        '## 配对规则对照','', '| 方法 | 配对局数 | 比规则多赢/多输 | 差值（百分点） | 保守 95% 区间 |','|---|---:|---:|---:|---|'])
    for r in methods:
        p=r['paired_vs_rule']
        if p['pairs']:
            lo,hi=p['interval95'];lines.append(f"| {r['method_id']} | {p['pairs']} | {p['n10']}/{p['n01']} | {p['difference']*100:+.2f} | [{lo*100:+.2f}, {hi*100:+.2f}] |")
    lines.extend(['','区间用 Bonferroni 组合两个 97.5% Clopper–Pearson 区间，只描述当前模型在测试开局上的差异，不能体现跨训练种子的波动。',
        '`no_observed_gain`：完整配对结果没有正点估计；`inconclusive`：尚未完成、缺少对照或区间未支持正差；`observed_gain_single_seed` 仍只限当前训练种子。','',
        '## 分难度结果','', '| 方法 | 蓝/红比例 | 成功/局数 | 规模宏平均 | 完整 |','|---|---:|---:|---:|---|'])
    for r in methods:
        for d in r['difficulties']: lines.append(f"| {r['method_id']} | {d['blue_red_ratio']:g} | {d['wins']}/{d['episodes']} | {pct(d['macro_success_rate'])} | {d['complete']} |")
    lines.extend(['','每格分子、分母、Wilson 区间和配对差：[comparison_cells.csv](comparison_cells.csv)。','',
        '## 决策与计算成本','', '| 方法 | 逐事件 p50（ms） | 逐事件 p95（ms） | 时延样本 | 正式测试真实步 | 正式测试规划模拟步 |','|---|---:|---:|---:|---:|---:|'])
    for r in methods: lines.append(f"| {r['method_id']} | {number(r['decision_p50_ms'])} | {number(r['decision_p95_ms'])} | {r['decision_samples']} | {r['real_environment_steps']} | {r['planner_sim_steps']} |")
    lines.extend(['','时延是运行时逐事件观测值，不能当作独占部署基准；未记录的时延留空。其余已记录训练、离线标签及验证/控制成本见 [costs.csv](costs.csv)。','',
        '串行部署测量：六任务工作进程全部退出后，在同一组预冻结状态上使用 CPU 单数值线程逐一测量。包含首次调用，排除环境恢复；外部系统负载只记录、未控制。',
        f"测量状态：{'已完成' if summary['serial_latency']['complete'] else '尚未全部完成'}。",'',
        '| 方法 | 串行 p50（ms） | 串行 p95（ms） | 串行样本 |', '|---|---:|---:|---:|'])
    for r in methods: lines.append(f"| {r['method_id']} | {number(r.get('serial_decision_p50_ms'))} | {number(r.get('serial_decision_p95_ms'))} | {r.get('serial_decision_samples',0)} |")
    if summary['serial_latency']['path']: lines.extend(['',f"[串行部署原始测量]({summary['serial_latency']['path']})。"])
    lines.extend(['','## 六路线当前判断','', '[逐路线的已观察、已排除与仍无法区分](task_analysis.md)。','',
        '| 路线 | 执行 | 仍无法区分 |','|---|---|---|'])
    for r in summary.get('task_analysis',[]): lines.append(f"| {r['task_id']} | {r['execution_status']} | {r['unresolved']} |")
    lines.extend(['',
        '## 诊断、曲线与复现范围','', '[同状态独立复核](diagnostics.csv)、[预算诊断](budget_diagnostics.csv)、[复现状态](reproduction_status.csv)、[完整 JSON](comparison.json)。'])
    for task,paths in summary['curves'].items():
        for label,path in paths.items(): lines.append(f'- {task} {label}：[查看]({path})')
    lines.extend(['','损失下降、部署率提高或候选标签更好，均不能替代原生任务成功率证据。预算结果采用独立 verification 分支；T4 教师复核比较的是该轮冻结续行策略，不应统一标成相对规则。',''])
    return '\n'.join(lines)


def update_ledger(run_dir,outputs):
    run_dir=Path(run_dir).resolve();config=read_json(run_dir/'shared/config_resolved.json')
    if run_dir.name!='v5_20260907_main' or config.get('smoke',True):
        return dict(updated=False,reason='not_authorized_formal_run')
    ledger=ROOT.parent/'实验记录.md'
    if not ledger.exists(): return dict(updated=False,reason='ledger_missing')
    start,end='<!-- V5_CURRENT_START -->','<!-- V5_CURRENT_END -->'
    frozen=read_json(run_dir/'shared/budget_manifest.json')
    lines=[f"更新于 {outputs.get('updated',datetime.now(timezone.utc).isoformat())}。",
        f"正式配置冻结：{frozen.get('frozen_at','尚未记录')}；训练种子：{config.get('seed')}。",
        '**六任务、全部预定末期测试及串行部署测量已完成。**' if outputs.get('complete') else '**训练、评估、诊断或串行部署测量尚未全部完成，当前不是最终有效性结论。**']
    for task in outputs.get('tasks',[]): lines.append(f"- {task['task_id']}：{task['execution_status']}，阶段 `{task['phase']}`。")
    lines.append(f"已有完整末期测试 {sum(r.get('complete',False) for r in outputs.get('methods',[]))}/14 臂。")
    if outputs.get('complete'): lines.append('有效性另按配对差和分难度结果判定，全部结论仍限于一个训练种子。')
    relative=os.path.relpath(run_dir/'final_report.md',ledger.parent).replace('\\','/')
    lines.append(f'[当前进度与结果报告]({relative})。')
    for _ in range(3):
        original=ledger.read_bytes();text=original.decode('utf-8')
        if text.count(start)!=1 or text.count(end)!=1 or text.index(end)<text.index(start):
            return dict(updated=False,reason='markers_missing_or_ambiguous')
        newline='\r\n' if '\r\n' in text else '\n'
        replacement=text[:text.index(start)+len(start)]+newline+newline.join(lines)+newline+text[text.index(end):]
        if ledger.read_bytes()!=original: continue
        atomic_text(ledger,replacement)
        return dict(updated=True,path=str(ledger))
    return dict(updated=False,reason='concurrent_user_edit')


def summarize(run_dir):
    run_dir=Path(run_dir).resolve();run_dir.mkdir(parents=True,exist_ok=True)
    config=read_json(run_dir/'shared/config_resolved.json');cache=read_json(run_dir/'.report_cache.json')
    found={}
    for path in run_dir.glob('T*/evaluations/*/test/*'):
        if not path.is_dir(): continue
        rows=episode_rows(path)
        if rows: found[rows[0].get('method_id',path.parents[1].name),rows[0].get('checkpoint_id',path.name)]=dict(
            rows=rows,directory=path,task_id=path.relative_to(run_dir).parts[0])
    all_methods=comparison_rows(found,config,cache.setdefault('latencies',{}))
    methods=[r for r in all_methods if MAIN.get(r['method_id'])==r['checkpoint_id']]
    supplementary=[r for r in all_methods if MAIN.get(r['method_id'])!=r['checkpoint_id']]
    deployment_latency=serial_latency(run_dir,methods)
    tasks=[task_status(run_dir/task) for task in TASK_METHODS]
    missing=[m for m in MAIN if not any(r['method_id']==m for r in methods)]
    diagnostics=diagnostic_rows(run_dir,cache.setdefault('diagnostics',{}))
    reproduction=reproduction_rows(run_dir,tasks,methods);costs=cost_rows(run_dir,tasks);curves={}
    for task in TASK_METHODS:
        directory=run_dir/task
        if not directory.exists(): continue
        plot_task(directory)
        images={label:(directory/file).relative_to(run_dir).as_posix() for label,file in (
            ('训练与验证曲线','training_curves.png'),('分难度训练成功率','success_by_difficulty.png'),('预算与独立复核曲线','budget_curves.png')) if (directory/file).exists()}
        if images: curves[task]=images
    summary=dict(schema='v5-report-v2',updated=datetime.now(timezone.utc).isoformat(),run_dir=str(run_dir),
        smoke=bool(config.get('smoke',False)),training_seeds=[config.get('seed')],methods=methods,
        supplementary_methods=supplementary,missing_methods=missing,expected_main_and_control_arms=len(MAIN),
        tasks=tasks,curves=curves,complete=not missing and all(r['complete'] for r in methods) and all(t['execution_status']=='complete' for t in tasks) and deployment_latency['complete'],
        task_execution_complete=all(t['execution_status']=='complete' for t in tasks),
        evaluation_complete=not missing and all(r['complete'] for r in methods),
        interpretation='Single training seed; execution completeness and effectiveness are separate.',
        costs=costs,reproduction_status=reproduction,diagnostic_rows=len(diagnostics),serial_latency=deployment_latency)
    summary['task_analysis']=task_analyses(run_dir,summary,diagnostics)
    atomic_json(run_dir/'comparison.json',summary);atomic_json(run_dir/'comparison_summary.json',summary)
    flat=[];cells=[];difficulties=[]
    for r in methods:
        values={k:v for k,v in r.items() if k not in ('cells','difficulties','paired_vs_rule')};p=r['paired_vs_rule']
        values.update(paired_episodes=p['pairs'],paired_difference=p['difference'],paired_lower95=p['interval95'][0],
            paired_upper95=p['interval95'][1],paired_extra_wins=p['n10'],paired_extra_losses=p['n01']);flat.append(values)
        cells.extend(dict(method_id=r['method_id'],checkpoint_id=r['checkpoint_id'],**c) for c in r['cells'])
        difficulties.extend(dict(method_id=r['method_id'],checkpoint_id=r['checkpoint_id'],**d) for d in r['difficulties'])
    write_csv(run_dir/'comparison.csv',flat, list(flat[0]) if flat else ['method_id','wins','episodes','complete','evidence_status'])
    write_csv(run_dir/'comparison_cells.csv',cells);write_csv(run_dir/'comparison_difficulty.csv',difficulties)
    write_csv(run_dir/'diagnostics.csv',diagnostics);write_csv(run_dir/'reproduction_status.csv',reproduction)
    write_csv(run_dir/'costs.csv',costs)
    write_csv(run_dir/'phase_costs.csv',[dict(r,task_id=task) for task in TASK_METHODS for r in records(run_dir/task/'phase_costs.jsonl')])
    write_csv(run_dir/'budget_diagnostics.csv',[r for task in TASK_METHODS for r in budget_rows(run_dir/task)])
    atomic_text(run_dir/'final_report.md',markdown(summary));atomic_json(run_dir/'.report_cache.json',cache)
    update_ledger(run_dir,summary)
    return summary
