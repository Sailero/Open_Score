"""Recompute HAD paper statistics offline and render publication figures.

Use --refresh-current to incorporate the completed main1013_base baselines.
Use --reuse-statistics to change figure layout without rereading episodes.csv.
No policy weights are loaded and no environment is executed by this script.
"""
import csv
import copy
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
HAD = ROOT / 'outputs/main/had'
OUT = HAD / 'figures'
SEEDS = (0, 1, 2)
SCENES = tuple(range(9000, 9300))
GROUPS = {
    'ID': ((4,4,2),(6,6,2),(8,8,2),(10,10,3)),
    'K-OOD': tuple((10,10,k) for k in (4,6,9,12)),
    'N-OOD': tuple((n,n,2) for n in (15,20,25,30,40,50)),
    'Joint-OOD': tuple((n,n,k) for n in (15,20,30) for k in (4,6)) + ((30,30,9),(30,30,12)),
}
CONFIGS = tuple(c for configs in GROUPS.values() for c in configs) + ((5,5,2),(10,10,2))
NATIVE = ['regir_nomem','regir_sg','regir_r0_sg','regir_r1_sg',
          'regir_norefil_sg','regir_last_sg','regir_fixed4_sg','regir_untied4_sg']
ALL_METHODS = NATIVE + ['refil','refil_matched','b2_qmix_atten','dcg','spectra','alma','transfqmix']
NAMES = {
    'regir_nomem':'LEAF', 'regir_sg':'LEAF–Memory query',
    'regir_r0_sg':'LEAF–Direct read', 'regir_r1_sg':'LEAF–One round',
    'regir_norefil_sg':'LEAF–No host', 'regir_last_sg':'LEAF–Last read',
    'regir_fixed4_sg':'LEAF–Fixed depth', 'regir_untied4_sg':'LEAF–Unshared rounds',
    'refil':'REFIL', 'refil_matched':'REFIL-Matched', 'b2_qmix_atten':'QMIX-Atten',
    'dcg':'DCG', 'spectra':'SPECTra', 'alma':'ALMA', 'transfqmix':'TransfQMIX',
    'rule_nv1':'Rule', 'random':'Random',
}
MECHANISM_METHODS = ('regir_nomem','regir_sg','refil')
MECHANISM_CONFIGS = ((10,10,2),(30,30,2),(50,50,2),(30,30,12))
EDGES = tuple(('regir_nomem', method, f'LEAF → {NAMES[method]}')
              for method in NATIVE if method != 'regir_nomem')
CORE_EDGES = tuple(edge for edge in EDGES
                   if edge[1] in ('regir_sg','regir_r0_sg','regir_r1_sg','regir_norefil_sg'))


def summary(values, unit='training_seed'):
    values = list(values)
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('A summary requires finite observations.')
    return {'mean':statistics.mean(values), 'sample_sd':statistics.stdev(values) if len(values)>1 else 0.,
            'per_seed':values if unit=='training_seed' else None, 'n':len(values), 'sd_unit':unit}


def decode(cell):
    return json.loads(cell) if cell else None


def mechanism(details, dup):
    out = {'scope':'formal 300 episodes per training seed and configuration',
        'first_damage_time_definition':'minimum non-null target time; no damage assigned 100 only for restricted-time mean and never counted as a CDF event',
        'hhi_definition':'sum((target_damage/D)^2), conditional on D>0',
        'summary':[], 'first_damage_cdf':[], 'leaf_vs_refil_directions':[], 'dup_summary':[]}
    for method in MECHANISM_METHODS:
        for cfg in MECHANISM_CONFIGS:
            seed_values, valid_n, cdfs = defaultdict(list), defaultdict(list), []
            for seed in SEEDS:
                episodes = list(details[method,seed,cfg].values())
                if len(episodes)!=300: raise ValueError('Incomplete behavior-summary cell.')
                values, events = defaultdict(list), []
                for e in episodes:
                    d=e['D']; times=[t for t in e['first_damage_step_by_target'] if t is not None]
                    first=min(times) if times else None; events.append(first)
                    values['collision_fraction'].append(e['collision_fraction'])
                    values['blue_intercepted_fraction'].append(e['blue_deaths_by_cause']['intercepted']/cfg[1])
                    values['red_left_fraction'].append(e['red_left']/cfg[0])
                    values['any_damage'].append(float(d>1e-9))
                    values['damaged_target_fraction'].append(sum(v>1e-9 for v in e['damage_by_target'])/cfg[2])
                    values['first_damage_restricted_time'].append(100 if first is None else first)
                    if d>0: values['damage_hhi'].append(sum((v/d)**2 for v in e['damage_by_target']))
                values['worst10pct_D']=sorted((e['D'] for e in episodes),reverse=True)[:30]
                values['worst10pct_reward']=[-v for v in values['worst10pct_D']]
                for key in ('collision_fraction','blue_intercepted_fraction','red_left_fraction','any_damage',
                            'damaged_target_fraction','first_damage_restricted_time','damage_hhi','worst10pct_D','worst10pct_reward'):
                    v=values[key]; seed_values[key].append(statistics.mean(v) if v else None); valid_n[key].append(len(v))
                cdfs.append([sum(t is not None and t<=step for t in events)/300 for step in range(101)])
            metrics={}
            for key, values in seed_values.items():
                nonnull=[v for v in values if v is not None]
                st=summary(nonnull) if nonnull else {'mean':None,'sample_sd':None,'n':0,'sd_unit':'training_seed'}
                st.update(per_seed=values,valid_episodes_per_seed=valid_n[key]); metrics[key]=st
            out['summary'].append({'method':method,'config':list(cfg),'metrics':metrics})
            out['first_damage_cdf'].append({'method':method,'config':list(cfg),'t':list(range(101)),
                'mean':[statistics.mean(v) for v in zip(*cdfs)],
                'sample_sd':[statistics.stdev(v) for v in zip(*cdfs)],'per_seed':cdfs,'n_per_seed':[300]*3})
    lookup={(r['method'],tuple(r['config'])):r for r in out['summary']}
    for cfg in MECHANISM_CONFIGS:
        for key in lookup['regir_nomem',cfg]['metrics']:
            a=lookup['regir_nomem',cfg]['metrics'][key]['per_seed']; b=lookup['refil',cfg]['metrics'][key]['per_seed']
            out['leaf_vs_refil_directions'].append({'config':list(cfg),'metric':key,
                'leaf_minus_refil_per_seed':[None if x is None or y is None else x-y for x,y in zip(a,b)]})
    for method in ('regir_nomem','refil'):
        for cfg in MECHANISM_CONFIGS[:3]:
            cells=[dup[method,seed,cfg] for seed in SEEDS]
            if any(len(cell)!=100 for cell in cells): raise ValueError('Incomplete common dup100 cell.')
            v=[[statistics.mean(e[i] for e in cell.values()) for cell in cells] for i in range(3)]
            out['dup_summary'].append({'method':method,'config':list(cfg),'n_per_seed':[100]*3,
                'reward':summary(-x for x in v[0]),'collision_fraction':summary(v[1]),'dup_pursuit_frac':summary(v[2])})
    cases=[]
    for cfg in (MECHANISM_CONFIGS[0],MECHANISM_CONFIGS[2]):
        episodes=details['regir_nomem',0,cfg]; scene=9000
        cases.append({'config':list(cfg),'training_seed':0,'episode_seed':scene,
            'selection':'preselected training seed 0, scene 9000, at the two representative scales',
            'evidence_type':'episode event summary, not a physical trajectory',
            'leaf':episodes[scene],'refil_same_scene':details['refil',0,cfg][scene]})
    return out,cases


def recompute(previous):
    final,rule,validation,details,dup=(defaultdict(dict) for _ in range(5))
    identities,phases,cfg_cache={},Counter(),{}
    validation_sources=defaultdict(lambda: {'source_versions':Counter(),'versions':Counter(),'arms':Counter()})
    with (HAD/'episodes.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            # Imported diagnostics retain their original protocol and may use
            # plain strings. Gate on the row version before decoding any fields.
            version = row.get('version', '').strip()
            if version.startswith('"'):
                version = decode(version)
            if version != 'main0928': continue
            if decode(row['env'])!='had': continue
            method,seed,phase,arm=decode(row['method']),int(row['seed']),decode(row['phase']),decode(row['arm'])
            phases[phase]+=1
            is_final=phase=='final_eval' and arm=='final' and method in ALL_METHODS
            is_rule=phase=='anchor' and method=='rule_nv1'
            is_val=phase=='train_eval' and method in ALL_METHODS
            is_dup=arm=='dup' and method in ('regir_nomem','refil')
            if not (is_final or is_rule or is_val or is_dup): continue
            if row['config'] not in cfg_cache:
                c=decode(row['config']); cfg_cache[row['config']]=tuple(int(c[k]) for k in ('N_R','N_B','K'))
            cfg=cfg_cache[row['config']]; scene=int(row['episode_seed']); d=float(row['D'])
            if not math.isfinite(d) or d<0: raise ValueError('Invalid damage metric.')
            if is_val:
                point=int(row['eval_point']); step=int(row['t_env'])
                if seed not in SEEDS or point not in range(1,51) or cfg not in GROUPS['ID'] or step<0:
                    raise ValueError(f'Invalid validation identity: {method}/{seed}/{point}/{cfg}')
                key=(method,seed,point,cfg)
                if scene in validation[key]: raise ValueError(f'Duplicate validation scene: {key}/{scene}')
                validation[key][scene]=(d,step)
                provenance=validation_sources[method]
                provenance['source_versions'][decode(row.get('source_version')) or decode(row.get('version')) or 'unregistered']+=1
                provenance['versions'][decode(row.get('version')) or 'unregistered']+=1
                provenance['arms'][arm]+=1
                continue
            deaths=decode(row['red_deaths_by_cause'])
            if not isinstance(deaths,dict) or deaths.get('friendly_collision') is None: raise ValueError('Missing collision count.')
            c=float(deaths['friendly_collision'])/cfg[0]
            if not math.isfinite(d) or d<0 or not 0<=c<=1: raise ValueError('Invalid outcome metric.')
            if is_final:
                key=(method,seed,cfg)
                if scene in final[key]: raise ValueError(f'Duplicate formal scene: {key}/{scene}')
                final[key][scene]=(d,c); identity=(decode(row['checkpoint_id']),int(row['t_env']))
                if key in identities and identities[key]!=identity: raise ValueError(f'Mixed checkpoint identities: {key}')
                identities[key]=identity
                if method in MECHANISM_METHODS and cfg in MECHANISM_CONFIGS:
                    targets=decode(row['damage_by_target']); first=decode(row['first_damage_step_by_target']); blue=decode(row['blue_deaths_by_cause'])
                    if len(targets)!=cfg[2] or len(first)!=cfg[2] or not math.isclose(sum(targets),d,rel_tol=1e-8,abs_tol=1e-8): raise ValueError('Invalid target summary.')
                    if any(t is not None and not 0<=t<=100 for t in first) or blue.get('intercepted') is None: raise ValueError('Invalid event/interception summary.')
                    details[key][scene]={'D':d,'reward':-d,'collision_fraction':c,'red_left':int(row['red_left']),
                        'blue_left':int(row['blue_left']),'ep_len':int(row['ep_len']),'damage_by_target':targets,
                        'first_damage_step_by_target':first,'red_deaths_by_cause':deaths,'blue_deaths_by_cause':blue,
                        'checkpoint_id':identity[0],'t_env':identity[1],'episode_seed':scene,'arm':arm}
            elif is_rule:
                if scene in rule[cfg]: raise ValueError('Duplicate Rule anchor scene.')
                rule[cfg][scene]=(d,c)
            elif is_dup:
                value=decode(row['dup_pursuit_frac'])
                if value is None or not 0<=value<=1: raise ValueError('Missing/invalid duplicate-pursuit value in dup arm.')
                dup[method,seed,cfg][scene]=(d,c,float(value))
    p={'schema':'had-paper1007-v3','created_at':datetime.now(timezone.utc).isoformat(),'source':'had/episodes.csv',
       'main_method':'regir_nomem','reward':'-D','collision_definition':'red_deaths_by_cause.friendly_collision / N_R',
       'learning_sd_unit':'three training-seed means after equal-weight configurations',
       'rule_sd_unit':'300 scene means after equal-weight configurations; not training SD',
       'group_order':list(GROUPS),'groups':{g:list(v) for g,v in GROUPS.items()},'methods':ALL_METHODS,
       'phase_rows':dict(phases),'config_summary':[],'rule_config':[],'group_summary':[],'per_seed_groups':[],
       'leaf_validation':[],'validation_summary':[],'validation_curves':[],'validation_sources':{},'paired_edges':[],
       'validation_protocol':{'phase':'train_eval','points':50,'training_seeds':list(SEEDS),
           'configs':[list(c) for c in GROUPS['ID']],'episodes_per_config':25,
           'reward':'-D','point_aggregation':'equal-weight configuration episode means within each training seed',
           'time_coordinate':'recorded t_env; maximum within a seed/point, then mean across the three seeds',
           'error_band':'sample SD across three training-seed point means; not within-point scene SD'},
       'baseline_learning_scope':'Original HAD baseline train_eval records recovered from main0928_supp/episodes_train_eval_had_baselines.csv and merged into had/episodes.csv. All learned curves use raw episode means and recorded t_env; no PDF-derived values or within-point scene SD enter these curves.'}
    for method in ALL_METHODS:
        for cfg in CONFIGS:
            cells=[final[method,seed,cfg] for seed in SEEDS]
            if any(set(cell)!=set(SCENES) for cell in cells): raise ValueError(f'Incomplete formal cell: {method}/{cfg}')
            ds=[statistics.mean(e[0] for e in cell.values()) for cell in cells]; cs=[statistics.mean(e[1] for e in cell.values()) for cell in cells]
            p['config_summary'].append({'method':method,'config':list(cfg),'D_per_seed':ds,'collision_fraction_per_seed':cs,
                'reward':summary(-v for v in ds),'collision_fraction':summary(cs),'collision_percent':summary(100*v for v in cs),
                'n_episodes_per_seed':[300]*3,'checkpoint_id_per_seed':[identities[method,s,cfg][0] for s in SEEDS],
                't_env_per_seed':[identities[method,s,cfg][1] for s in SEEDS]})
        for group,configs in GROUPS.items():
            ds=[statistics.mean(statistics.mean(e[0] for e in final[method,seed,cfg].values()) for cfg in configs) for seed in SEEDS]
            cs=[statistics.mean(statistics.mean(e[1] for e in final[method,seed,cfg].values()) for cfg in configs) for seed in SEEDS]
            p['group_summary'].append({'method':method,'group':group,'n_configs':len(configs),'D_per_seed':ds,
                'collision_fraction_per_seed':cs,'reward':summary(-v for v in ds),'collision_fraction':summary(cs),'collision_percent':summary(100*v for v in cs)})
            p['per_seed_groups'].extend({'method':method,'group':group,'seed':s,'reward':-ds[s],'collision_fraction':cs[s]} for s in SEEDS)
    for cfg in CONFIGS:
        if set(rule[cfg])!=set(SCENES): raise ValueError(f'Incomplete Rule cell: {cfg}')
        v=[rule[cfg][scene] for scene in SCENES]
        p['rule_config'].append({'method':'rule_nv1','config':list(cfg),'reward':summary((-e[0] for e in v),'scene'),
            'collision_fraction':summary((e[1] for e in v),'scene'),'collision_percent':summary((100*e[1] for e in v),'scene')})
    for group,configs in GROUPS.items():
        ds=[statistics.mean(rule[cfg][scene][0] for cfg in configs) for scene in SCENES]
        cs=[statistics.mean(rule[cfg][scene][1] for cfg in configs) for scene in SCENES]
        entry={'method':'rule_nv1','group':group,'n_configs':len(configs),'reward':summary((-v for v in ds),'scene'),
            'collision_fraction':summary(cs,'scene'),'collision_percent':summary((100*v for v in cs),'scene')}
        p['group_summary'].append(entry)
        if group=='ID': p['rule_ID_reference']=entry
    for method in ALL_METHODS:
        method_points=[]
        for point in range(1,51):
            seed_points=[]
            for seed in SEEDS:
                cells=[validation[method,seed,point,cfg] for cfg in GROUPS['ID']]
                if any(len(cell)!=25 for cell in cells): raise ValueError(f'Incomplete validation point: {method}/{seed}/{point}')
                rewards=[-statistics.mean(e[0] for e in cell.values()) for cell in cells]
                steps=[e[1] for cell in cells for e in cell.values()]
                entry={'method':method,'seed':seed,'eval_point':point,'t_env':max(steps),
                    't_env_minmax':[min(steps),max(steps)],'reward':statistics.mean(rewards),
                    'reward_per_config':rewards,'n':100,'n_per_config':[25]*4}
                p['validation_summary'].append(entry); seed_points.append(entry); method_points.append(entry)
                if method=='regir_nomem':
                    p['leaf_validation'].append({k:entry[k] for k in ('seed','eval_point','t_env','reward','n','n_per_config')})
            p['validation_curves'].append({'method':method,'eval_point':point,
                't_env_mean':statistics.mean(r['t_env'] for r in seed_points),
                't_env_per_seed':[r['t_env'] for r in seed_points],
                'reward':summary(r['reward'] for r in seed_points),'n_episodes_per_seed':[100]*3})
        provenance=validation_sources[method]
        p['validation_sources'][method]={k:dict(v) for k,v in provenance.items()}
        p['validation_sources'][method].update(raw_source='had/episodes.csv',n_raw_episodes=15000,
            n_seed_points=len(method_points),n_curve_points=50,training_seeds=list(SEEDS),
            t_env_minmax=[min(r['t_env_minmax'][0] for r in method_points),max(r['t_env'] for r in method_points)])
    lookup={(r['method'],r['group']):r for r in p['group_summary']}
    for left,right,label in EDGES:
        for group in GROUPS:
            delta=[a-b for a,b in zip(lookup[left,group]['D_per_seed'],lookup[right,group]['D_per_seed'])]
            p['paired_edges'].append({'left':left,'right':right,'label':label,'group':group,
                'delta_reward_right_minus_left':summary(delta),'positive_seeds':sum(v>1e-12 for v in delta),
                'negative_seeds':sum(v< -1e-12 for v in delta),
                'scope':'composite host-removal comparison' if right=='regir_norefil_sg' else 'saved-configuration comparison'})
    p['mechanism_analysis'],p['selected_cases']=mechanism(details,dup)
    p['formal_coverage']={'learned_methods':len(ALL_METHODS),'training_seeds':3,'configs':24,'episodes_per_cell':300,
        'learned_rows':sum(map(len,final.values())),'rule_rows':sum(map(len,rule.values())),
        'leaf_validation_rows':15000,'validation_methods':len(ALL_METHODS),
        'validation_rows':sum(map(len,validation.values())),'auxiliary_rollouts_included_in_formal_means':False}
    for key in ('auxiliary1007','history_interventions','random_config','random_group_summary','random_scope'):
        if key in previous: p[key]=previous[key]
    return p


def cached_statistics(aggregates):
    cache = aggregates.get('paper1007')
    if cache is None:
        cache = aggregates.get('sources',{}).get('main0928',{}).get('paper1007')
    if cache is None:
        raise ValueError('No saved paper1007 statistics found in the current HAD aggregate.')
    return cache


def refresh_current(previous):
    """Replace only the two rerun baselines, preserving the archived comparison."""
    methods = ('spectra', 'transfqmix')
    version = 'main1013_base'
    protocol = 'main1013_base_last_action_v1'
    configs = CONFIGS
    pools = {'standard': tuple(range(9000, 9300)),
             'confirmation': tuple(range(110000, 110300))}
    evaluations, validation = defaultdict(dict), defaultdict(dict)
    identities, sources, counts = {}, defaultdict(Counter), Counter()

    def decoded(row, key):
        value = row.get(key)
        if value in (None, ''):
            return None
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value

    with (HAD / 'episodes.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if decoded(row, 'version') != version and decoded(row, 'source_version') != version:
                continue
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Truncated main1013_base episode record.')
            method, phase = decoded(row, 'method'), decoded(row, 'phase')
            if method not in methods or phase not in ('train_eval', 'final_eval'):
                continue
            if decoded(row, 'env') != 'had':
                raise ValueError('Unexpected main1013_base environment.')
            seed, scene = int(row['seed']), int(row['episode_seed'])
            c = decoded(row, 'config')
            cfg = tuple(int(c[key]) for key in ('N_R', 'N_B', 'K'))
            damage, step = float(row['D']), int(row['t_env'])
            if seed not in SEEDS or not math.isfinite(damage) or damage < 0:
                raise ValueError(f'Invalid current baseline observation: {method}/{seed}/{cfg}')
            sources[method][decoded(row, 'source_version') or decoded(row, 'version')] += 1
            counts[phase] += 1
            if phase == 'train_eval':
                point = int(row['eval_point'])
                if cfg not in GROUPS['ID'] or point not in range(1, 51) or not 0 <= step <= 1000000:
                    raise ValueError(f'Invalid current validation identity: {method}/{seed}/{point}/{cfg}')
                index = GROUPS['ID'].index(cfg)
                if scene not in range(9500 + 25 * index, 9525 + 25 * index):
                    raise ValueError('Current validation scene is outside its frozen cell.')
                cell = validation[method, seed, point, cfg]
                if scene in cell:
                    raise ValueError('Duplicate current validation observation.')
                cell[scene] = (damage, step)
                continue
            arm, artifact = decoded(row, 'arm'), decoded(row, 'checkpoint_id')
            if (arm not in pools or cfg not in configs or scene not in pools[arm]
                    or decoded(row, 'protocol') != protocol or not artifact
                    or step != 1000000 or int(row['eval_point']) != 50):
                raise ValueError(f'Invalid current formal identity: {method}/{seed}/{arm}/{cfg}')
            identity_key = (method, seed)
            identity = (artifact, step, protocol)
            if identity_key in identities and identities[identity_key] != identity:
                raise ValueError(f'Mixed current checkpoint identities: {identity_key}')
            identities[identity_key] = identity
            cell = evaluations[method, arm, seed, cfg]
            if scene in cell:
                raise ValueError('Duplicate current formal observation.')
            cell[scene] = damage

    p = copy.deepcopy(previous)
    p.update(schema='had-paper-current-v1', created_at=datetime.now(timezone.utc).isoformat(),
             source='had/episodes.csv', historical_statistics='paper1007',
             current_baseline_version=version,
             current_baseline_protocol=protocol)
    for key in ('config_summary', 'group_summary', 'per_seed_groups',
                'validation_summary', 'validation_curves'):
        p[key] = [row for row in p[key] if row.get('method') not in methods]
    confirmation = {'schema': 'had-paper-current-confirmation-v1',
                    'source': 'had/episodes.csv', 'source_version': version,
                    'protocol': protocol, 'arm': 'confirmation',
                    'scene_seeds': [110000, 110299], 'methods': list(methods),
                    'config_summary': [], 'group_summary': [], 'per_seed_groups': []}
    current_groups = dict(GROUPS, All24=configs)
    p.setdefault('method_sources', {})
    for method in methods:
        for arm, scene_bank in pools.items():
            target = p if arm == 'standard' else confirmation
            for cfg in configs:
                cells = [evaluations[method, arm, seed, cfg] for seed in SEEDS]
                if any(set(cell) != set(scene_bank) for cell in cells):
                    raise ValueError(f'Incomplete current formal cell: {method}/{arm}/{cfg}')
                ds = [statistics.mean(cell.values()) for cell in cells]
                target['config_summary'].append({
                    'method': method, 'config': list(cfg), 'D_per_seed': ds,
                    'damage': summary(ds), 'reward': summary(-d for d in ds),
                    'collision_fraction_per_seed': None, 'collision_fraction': None,
                    'collision_percent': None, 'interception_fraction': None,
                    'n_episodes_per_seed': [300] * 3,
                    'checkpoint_id_per_seed': [identities[method, seed][0] for seed in SEEDS],
                    't_env_per_seed': [1000000] * 3,
                    'source_version': version, 'protocol': protocol, 'arm': arm})
            for group, cells in current_groups.items():
                ds = [statistics.mean(statistics.mean(evaluations[method, arm, seed, cfg].values())
                                      for cfg in cells) for seed in SEEDS]
                target['group_summary'].append({
                    'method': method, 'group': group, 'n_configs': len(cells),
                    'D_per_seed': ds, 'damage': summary(ds), 'reward': summary(-d for d in ds),
                    'collision_fraction_per_seed': None, 'collision_fraction': None,
                    'collision_percent': None, 'interception_fraction': None,
                    'source_version': version, 'protocol': protocol, 'arm': arm})
                target['per_seed_groups'].extend({
                    'method': method, 'group': group, 'seed': seed,
                    'D': ds[seed], 'reward': -ds[seed], 'collision_fraction': None,
                    'source_version': version, 'protocol': protocol, 'arm': arm}
                    for seed in SEEDS)
        method_points = []
        for point in range(1, 51):
            seed_points = []
            for seed in SEEDS:
                cells = [validation[method, seed, point, cfg] for cfg in GROUPS['ID']]
                if any(len(cell) != 25 for cell in cells):
                    raise ValueError(f'Incomplete current validation point: {method}/{seed}/{point}')
                rewards = [-statistics.mean(d for d, _ in cell.values()) for cell in cells]
                steps = [step for cell in cells for _, step in cell.values()]
                entry = {'method': method, 'seed': seed, 'eval_point': point,
                         't_env': max(steps), 't_env_minmax': [min(steps), max(steps)],
                         'reward': statistics.mean(rewards), 'reward_per_config': rewards,
                         'n': 100, 'n_per_config': [25] * 4, 'source_version': version}
                p['validation_summary'].append(entry)
                method_points.append(entry)
                seed_points.append(entry)
            p['validation_curves'].append({
                'method': method, 'eval_point': point,
                't_env_mean': statistics.mean(row['t_env'] for row in seed_points),
                't_env_per_seed': [row['t_env'] for row in seed_points],
                'reward': summary(row['reward'] for row in seed_points),
                'n_episodes_per_seed': [100] * 3, 'source_version': version})
        for seed in SEEDS:
            last = next(row for row in method_points if row['seed'] == seed and row['eval_point'] == 50)
            if last['t_env'] != 1000000:
                raise ValueError(f'Incomplete current training budget: {method}/{seed}')
        p['validation_sources'][method] = {
            'source_versions': {version: 15000}, 'versions': {version: 15000},
            'arms': {'unregistered': 15000}, 'raw_source': 'had/episodes.csv',
            'n_raw_episodes': 15000, 'n_seed_points': 150, 'n_curve_points': 50,
            'training_seeds': list(SEEDS),
            't_env_minmax': [min(row['t_env_minmax'][0] for row in method_points), 1000000]}
        p['method_sources'][method] = {
            'source_version': version, 'protocol': protocol, 'entity_input_width': 19,
            'training_seeds': list(SEEDS), 'checkpoint': 'final', 't_env': 1000000,
            'formal_arm': 'standard', 'confirmation_arm': 'confirmation',
            'formal_rows': 21600, 'confirmation_rows': 21600, 'validation_rows': 15000,
            'missing_formal_metrics': ['friendly_collision', 'interception'],
            'alignment': 'entity input and task/sampling controls; native optimizer and TD settings retained'}
    confirmation['coverage'] = {'methods': 2, 'training_seeds': 3, 'configs': 24,
                                'episodes_per_cell': 300, 'rows': 43200}
    p['confirmation'] = confirmation
    p['formal_coverage'].update(
        learned_rows=sum(sum(row['n_episodes_per_seed']) for row in p['config_summary']),
        current_baseline_rows=43200, current_baseline_validation_rows=30000,
        current_baseline_confirmation_rows=43200,
        current_baseline_configs=24)
    p['current_baseline_coverage'] = {
        'source_version': version, 'phase_rows': dict(counts),
        'source_rows_per_method': {method: dict(sources[method]) for method in methods},
        'total_rows': sum(counts.values()), 'expected_total_rows': 116400,
        'formal_rows': 43200, 'confirmation_rows': 43200, 'validation_rows': 30000}
    p['baseline_learning_scope'] = (
        'Current SPECTra and TransfQMix curves use main1013_base raw train_eval rows; '
        'other learned curves retain the historical paper1007 statistics. '
        'Each point is an equal-weight ID configuration mean, with sample SD across three training seeds.')
    return p


def active_statistics(cache):
    """Filter legacy caches in memory without rewriting aggregate provenance."""
    allowed = set(ALL_METHODS) | {'rule_nv1','random'}

    def keep(value):
        if isinstance(value,dict):
            return all(value.get(key) in allowed for key in ('method','left','right') if key in value)
        return not isinstance(value,str) or not value.startswith('regir_') or value in allowed

    def filtered(value):
        if isinstance(value,list):
            return [filtered(item) for item in value if keep(item)]
        if isinstance(value,dict):
            return {key:filtered(item) for key,item in value.items()
                    if not key.startswith('regir_') or key in allowed}
        return value

    p = filtered(cache)
    p['methods'] = list(ALL_METHODS)
    lookup = {(r['method'],r['group']):r for r in p['group_summary']}
    p['paired_edges'] = []
    for left,right,label in EDGES:
        for group in GROUPS:
            delta = [a-b for a,b in zip(lookup[left,group]['D_per_seed'],lookup[right,group]['D_per_seed'])]
            p['paired_edges'].append({'left':left,'right':right,'label':label,'group':group,
                'delta_reward_right_minus_left':summary(delta),
                'positive_seeds':sum(v>1e-12 for v in delta),'negative_seeds':sum(v< -1e-12 for v in delta),
                'scope':'composite host-removal comparison' if right=='regir_norefil_sg' else 'saved-configuration comparison'})
    p['formal_coverage'].update(learned_methods=len(ALL_METHODS),validation_methods=len(ALL_METHODS),
        learned_rows=sum(sum(r['n_episodes_per_seed']) for r in p['config_summary']),
        validation_rows=sum(r['n'] for r in p['validation_summary']))
    return p


AGGREGATES = json.loads((HAD/'aggregates.json').read_text(encoding='utf-8'))
if '--refresh-current' in sys.argv:
    P = active_statistics(refresh_current(cached_statistics(AGGREGATES)))
    AGGREGATES['paper_current'] = P
    (HAD/'aggregates.json').write_text(json.dumps(AGGREGATES,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(P['current_baseline_coverage']))
elif 'paper_current' in AGGREGATES:
    P = active_statistics(AGGREGATES['paper_current'])
elif '--reuse-statistics' in sys.argv:
    P=active_statistics(cached_statistics(AGGREGATES))
else:
    P=active_statistics(recompute(cached_statistics(AGGREGATES)))
    AGGREGATES['paper1007']=P
    (HAD/'aggregates.json').write_text(json.dumps(AGGREGATES,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(P['formal_coverage']))
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
    'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'savefig.dpi': 220})
WIDTH = 7.16
METHODS = ['regir_nomem', 'refil', 'b2_qmix_atten', 'dcg', 'spectra', 'alma', 'transfqmix', 'rule_nv1']
ABLATION_METHODS = ['regir_nomem','regir_sg','regir_norefil_sg','regir_r0_sg','regir_r1_sg']
COLORS = dict(zip(METHODS, ['#087f8c', '#df8f44', '#4775b4', '#9768aa', '#be5962', '#799d54', '#ad784f', '#6c727a']))
COLORS.update({'regir_sg':'#7388b7','regir_norefil_sg':'#c79532',
               'regir_r0_sg':'#ba5671','regir_r1_sg':'#745399'})
CONFIG = {(r['method'], tuple(r['config'])): r for r in P['config_summary']}
CONFIG.update({('rule_nv1', tuple(r['config'])): r for r in P['rule_config']})
GROUP = {(r['method'], r['group']): r for r in P['group_summary']}
VALIDATION = {(r['method'],r['eval_point']):r for r in P['validation_curves']}

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', facecolor='white')
    plt.close(fig)

def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e5e8eb', linewidth=.5)

def curve(ax, method, configs, xs):
    rows = [CONFIG[method, tuple(c)] for c in configs]
    y = np.array([r['reward']['mean'] for r in rows])
    sd = np.array([r['reward']['sample_sd'] for r in rows])
    color = COLORS[method]
    if method != 'rule_nv1':
        ax.fill_between(xs, y-sd, y+sd, color=color, alpha=.075, linewidth=0)
        for x, row in zip(xs, rows):
            ax.scatter([x]*3, -np.array(row['D_per_seed']), s=5, color=color, alpha=.22, linewidths=0)
    ax.plot(xs, y, color=color, marker='o', ms=2.4, lw=1.8 if method=='regir_nomem' else 1.05,
            ls='--' if method=='rule_nv1' else '-', label=NAMES[method])

def validation_curve(ax, method):
    rows=[VALIDATION[method,point] for point in range(1,51)]
    xs=np.array([r['t_env_mean'] for r in rows])/1e6
    mean=np.array([r['reward']['mean'] for r in rows])
    sd=np.array([r['reward']['sample_sd'] for r in rows])
    color=COLORS[method]
    ax.fill_between(xs,mean-sd,mean+sd,color=color,alpha=.10,linewidth=0)
    ax.plot(xs,mean,color=color,lw=1.8 if method=='regir_nomem' else 1.1,
            label=NAMES[method],zorder=4 if method=='regir_nomem' else 3)


def joint_heatmap(fig, ax, methods):
    joint=list(GROUPS['Joint-OOD'])
    values=np.array([[CONFIG[m,c]['reward']['mean'] for c in joint] for m in methods])
    im=ax.imshow(values,aspect='auto',cmap='viridis',vmin=np.floor(values.min()),vmax=0,
                 interpolation='none')
    ax.set(title='(d) Joint extrapolation',xticks=np.arange(8),
           xticklabels=[f'{c[0]}/{c[2]}' for c in joint],yticks=np.arange(len(methods)),
           yticklabels=[NAMES[m].replace('–','\n–') for m in methods],xlabel='Vehicles / targets ($N/K$)')
    ax.tick_params(axis='x',rotation=55,pad=1)
    ax.tick_params(axis='y',length=0,pad=3)
    ax.get_yticklabels()[0].set_weight('bold')
    for i,row in enumerate(values):
        for j,value in enumerate(row):
            rgb=im.cmap(im.norm(value))[:3]
            lightness=sum(a*b for a,b in zip(rgb,(.2126,.7152,.0722)))
            ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=8,
                    color='black' if lightness>.50 else 'white')
    fig.colorbar(im,ax=ax,fraction=.047,pad=.035,label=r'$R$ $\uparrow$')


def legend_handles(methods):
    return [Line2D([0],[0],color=COLORS[m],lw=1.8 if m=='regir_nomem' else 1.1,
                   ls='--' if m=='rule_nv1' else '-',marker='o',ms=3,label=NAMES[m])
            for m in methods]


# Four-panel comparison. Joint cells show means; the table reports dispersion.
fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.4))
fig.subplots_adjust(left=.095, right=.918, bottom=.215, top=.945, hspace=.47, wspace=.43)
ax = axes[0, 0]
for method in METHODS[:-1]: validation_curve(ax,method)
ax.axhline(P['rule_ID_reference']['reward']['mean'],color=COLORS['rule_nv1'],ls='--',lw=1)
ax.set(title='(a) Training validation',xlabel='Environment steps (millions)',
       ylabel=r'Reward $R=-D$ $\uparrow$',xlim=(0,1.01),xticks=[0,.25,.5,.75,1])
ax.text(.04,.05,'Rule: final ID reference',transform=ax.transAxes,fontsize=8)
style(ax)
ax = axes[0,1]
ks=[4,6,9,12]
for method in METHODS: curve(ax,method,[(10,10,k) for k in ks],ks)
ax.set(title='(b) Target extrapolation',xlabel='Targets $K$ (10 vehicles per team)',
       ylabel=r'Reward $R$ $\uparrow$',xticks=ks)
style(ax)
ax=axes[1,0]
ns=[15,20,25,30,40,50]
for method in METHODS: curve(ax,method,[(n,n,2) for n in ns],ns)
ax.set(title='(c) Vehicle extrapolation',xlabel='Vehicles per team $N$ ($K=2$)',
       ylabel=r'Reward $R$ $\uparrow$',xticks=[15,25,40,50])
style(ax)
joint_heatmap(fig,axes[1,1],METHODS)
fig.legend(handles=legend_handles(METHODS),loc='lower center',bbox_to_anchor=(.52,.017),
           ncol=4,frameon=False,columnspacing=1.2,handlelength=1.8)
save(fig,'fig_had_generalization')

# Core saved configurations: original paired-edge statistics remain in the aggregate.
fig,axes=plt.subplots(2,2,figsize=(WIDTH,6.4))
fig.subplots_adjust(left=.095,right=.918,bottom=.195,top=.945,hspace=.47,wspace=.65)
ax=axes[0,0]
for method in ABLATION_METHODS: validation_curve(ax,method)
ax.set(title='(a) Training validation',xlabel='Environment steps (millions)',
       ylabel=r'Reward $R=-D$ $\uparrow$',xlim=(0,1.01),xticks=[0,.25,.5,.75,1])
style(ax)
ax=axes[0,1]
for method in ABLATION_METHODS: curve(ax,method,[(10,10,k) for k in ks],ks)
ax.set(title='(b) Target extrapolation',xlabel='Targets $K$ (10 vehicles per team)',
       ylabel=r'Reward $R$ $\uparrow$',xticks=ks)
style(ax)
ax=axes[1,0]
for method in ABLATION_METHODS: curve(ax,method,[(n,n,2) for n in ns],ns)
ax.set(title='(c) Vehicle extrapolation',xlabel='Vehicles per team $N$ ($K=2$)',
       ylabel=r'Reward $R$ $\uparrow$',xticks=[15,25,40,50])
style(ax)
joint_heatmap(fig,axes[1,1],ABLATION_METHODS)
fig.legend(handles=legend_handles(ABLATION_METHODS),loc='lower center',bbox_to_anchor=(.52,.025),
           ncol=2,frameon=False,columnspacing=1.5,handlelength=1.8)
save(fig,'fig_had_ablation')

M=P['mechanism_analysis']
MC={(r['method'],tuple(r['config'])):r for r in M['summary']}
fig,axes=plt.subplots(2,2,figsize=(WIDTH,4.9))
fig.subplots_adjust(left=.10,right=.985,bottom=.19,top=.945,hspace=.43,wspace=.31)
for ax,key,title,ylabel in [(axes[0,0],'collision_fraction','(a) Friendly-collision deaths','Fraction of initial defenders (%)'),(axes[0,1],'blue_intercepted_fraction','(b) Attacker interceptions','Fraction of initial attackers (%)')]:
    for m in ['regir_nomem','regir_sg','refil']:
        stats=[MC[m,(n,n,2)]['metrics'][key] for n in [10,30,50]]
        mean=np.array([x['mean'] for x in stats])*100; sd=np.array([x['sample_sd'] for x in stats])*100
        ax.errorbar([10,30,50],mean,yerr=sd,color=COLORS[m],marker='o',ms=3,capsize=2,lw=1.2,label=NAMES[m])
        for n,st in zip([10,30,50],stats):
            ax.scatter([n]*3,np.array(st['per_seed'])*100,s=7,color=COLORS[m],alpha=.3,linewidths=0)
    ax.set(title=title,xlabel='Vehicles per team $N$ ($K=2$)',ylabel=ylabel,xticks=[10,30,50]); style(ax)
for ax,c,title in [(axes[1,0],(50,50,2),'(c) First target damage: $N=50, K=2$'),(axes[1,1],(30,30,12),'(d) First target damage: $N=30, K=12$')]:
    for m in ['regir_nomem','regir_sg','refil']:
        r=next(r for r in M['first_damage_cdf'] if r['method']==m and tuple(r['config'])==c)
        mean=np.array(r['mean']); sd=np.array(r['sample_sd']); t=r['t']
        ax.fill_between(t,np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=COLORS[m],alpha=.1,lw=0)
        ax.step(t,mean,where='post',color=COLORS[m],lw=1.3)
    ax.set(title=title,xlabel='Physical step $t$',ylabel=r'$\Pr(T_{\rm first}\leq t)$',xlim=(0,100),ylim=(0,1),xticks=[0,25,50,75,100]); style(ax)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.02),ncol=3,frameon=False)
save(fig,'fig_had_learning_behavior')

# The frozen LEAF depth matrix is independent of the archived M1–M5 comparison.
depth_analysis = json.loads((HAD.parent/'summary.json').read_text(encoding='utf-8')).get('nomem_depth_analysis')
if depth_analysis is not None:
    fig,axes = plt.subplots(2,3,figsize=(13,7.5))
    fig.subplots_adjust(left=.065,right=.985,bottom=.075,top=.82,hspace=.34,wspace=.27)
    fig.suptitle('LEAF: matched 24 configurations, 300 scenes per seed and round\n'
                 'Error bars: SD across three training seeds; shaded area: beyond training depth',
                 fontsize=13,y=.98)
    for ax,group in zip(axes.flat,('All24','ID','K-OOD','N-OOD','Joint-OOD','Extra')):
        rows = [depth_analysis['groups'][group]['depths'][str(r)] for r in range(1,7)]
        ax.axvspan(4.5,6.3,color='#f2c48c',alpha=.18,zorder=0)
        ax.errorbar(range(1,7),[r['mean'] for r in rows],yerr=[r['seed_sd'] for r in rows],
                    color=COLORS['regir_nomem'],marker='o',capsize=3,lw=1.5)
        ax.set(title=group,xlabel='Fixed execution rounds',ylabel='D (lower is better)',
               xticks=range(1,7),xticklabels=[f'R{r}' for r in range(1,7)],xlim=(.7,6.3))
        style(ax)
    fig.savefig(OUT/'loop_nomem_depth.png',facecolor='white',dpi=170)
    plt.close(fig)

print('Rendered HAD comparison, training/OOD ablation, behavior and saved LEAF depth figures.')
