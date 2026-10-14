"""Normal frozen-depth evaluation and factual fusion analysis for LEAF v3.

PowerShell, from E:/Code/Open_Score:
    $env:PYTHONPATH='E:/Code/SkyHAD'
    & 'D:/Software/Anaconda/envs/torch310/python.exe' Papers/v3/analysis/mechanism_leaf1008.py
Append --summarize-only after the registered 450 cells are complete to regenerate
offline statistics and figures. No training or destructive forward interventions.
Matching previously saved final episodes are reused with explicit provenance;
full L4 read vectors and physical states are recaptured on the normal policy.
All output is confined to Papers/v3.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import base64
import gzip
import json
from pathlib import Path
import random
import time

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
SOURCE = REPO / 'Open-SCORE'
OUTPUT = SOURCE / 'outputs/main'
sys.path.insert(0, str(SOURCE / 'scripts'))
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from aux_leaf1007 import save, packed, replay_check, physical_state
from open_score.algos import load_policy
from open_score.eval.anchors import BLUE_STRATEGY
from open_score.rules import register_end_to_end_policy, run_episode
import had_env
from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL

OLD = OUTPUT / 'had/regir_nomem/mechanisms/results.json.gz'
DATA = PAPER / 'data/mechanism1008.json.gz'
CONFIGS = ((10, 10, 2), (30, 30, 2), (30, 30, 12))
SCENES = tuple(range(9000, 9010))
DEPTHS = ('L1', 'L2', 'L3', 'L4')
SCHEMA = 'leaf-normal-depth-factual1008-v3'


def unpack(x):
    return np.frombuffer(base64.b64decode(x['base64']), dtype=x['dtype']).reshape(x['shape'])


def ek(e):
    return e['method'], int(e['training_seed']), tuple(e['config']), int(e['episode_seed']), e['arm']


def expected_keys():
    return {(m, s, c, e, a) for m in ('regir_nomem', 'refil') for s in range(3)
            for c in CONFIGS for e in SCENES for a in (DEPTHS if m == 'regir_nomem' else ('standard',))}


def pressure(state, n_blue, scale):
    blues = np.asarray([e.position[:2] for e in state.blue if e.health > 0], dtype=float)
    targets = np.asarray([e.position[:2] for e in state.targets], dtype=float)
    if not len(blues) or not len(targets): return 0.0
    nearest = np.linalg.norm(blues[:, None, :] - targets[None, :, :], axis=-1).min(axis=1)
    return float(np.clip(1 - nearest / scale, 0, 1).sum() / n_blue)


class Control:
    """Observe the normal read-score inputs; hooks never change model output."""
    def __init__(self, policy, method, depth=4):
        self.policy, self.method, self.depth = policy, method, depth
        self.branch = policy.mac.agent.global_net if method == 'regir_nomem' else None
        self.handle = self.branch.read_score.register_forward_hook(self._score) if self.branch else None
        self.capture = False
        self.scale = float(getattr(policy.args, 'shaping_range', 4000.0))

    def _score(self, module, inputs, output):
        if self.capture:
            self.reads.append(inputs[0].detach().clone())
            self.scores.append(output.detach().clone())

    def reset(self):
        self.policy.reset()
        if self.branch is not None:
            self.policy.set_eval_depth(self.depth)
            if self.branch.readout_override is not None or getattr(self.branch.args, 'global_read_h0', False):
                raise RuntimeError('Nonstandard readout in registered normal-depth checkpoint')
        self.decisions, self.initial = [], None

    @torch.no_grad()
    def act(self, state, side, action_ids):
        if self.initial is None: self.initial = physical_state(state)
        self.capture = self.branch is not None and self.depth == 4
        self.reads, self.scores = [], []
        actions = self.policy.act(state, side, action_ids)
        self.capture = False
        if self.branch is None or self.depth != 4: return actions
        active = [i for i, e in enumerate(state.red) if e.alive]
        if not active: return actions
        if len(self.reads) != 4 or len(self.scores) != 4:
            raise RuntimeError('Normal L4 did not expose exactly four intermediate reads')
        reads = torch.stack(self.reads, dim=1)[active]
        alpha = torch.softmax(torch.cat(self.scores, dim=-1), dim=-1)[active]
        self.decisions.append({'step': int(state.step),
            'active_defender_ids': [int(state.red[i].id) for i in active], 'n_active': len(active),
            'physical_state': physical_state(state), 'actions': {str(k): int(v) for k, v in actions.items()},
            'pressure': pressure(state, len(state.blue), self.scale),
            'reads': packed(reads), 'alpha': packed(alpha)})
        return actions


def stats(values):
    x = np.asarray(values, dtype=float)
    return {'mean': float(x.mean()), 'sample_sd': float(x.std(ddof=1)), 'per_seed': x.tolist()}


def vector_stats(values):
    x = np.asarray(values, dtype=float)
    return {'mean': x.mean(0).tolist(), 'sample_sd': x.std(0, ddof=1).tolist(), 'per_seed': x.tolist()}


def metrics(e):
    s, c = e['episode_summary'], e['config']
    return {'R': -float(s['D']),
        'collision_mortality': float(s['red_deaths_by_cause']['friendly_collision']) / c[0],
        'interception_fraction': float(s['blue_deaths_by_cause']['intercepted']) / c[1],
        'target_damage_incidence': float(s['D'] > 0)}


def summarize(data):
    eps = {ek(e): e for e in data['episodes']}
    if len(eps) != len(data['episodes']) or not set(eps) <= expected_keys():
        raise RuntimeError('Duplicate or unregistered episode cell')
    rows, groups, paired = [], [], []
    for cfg in CONFIGS:
        for method, arms in (('regir_nomem', DEPTHS), ('refil', ('standard',))):
            for arm in arms:
                local = []
                for seed in range(3):
                    keys = [(method, seed, cfg, scene, arm) for scene in SCENES]
                    if not all(k in eps for k in keys): continue
                    vals = [metrics(eps[k]) for k in keys]
                    row = {'method': method, 'arm': arm, 'config': list(cfg), 'training_seed': seed,
                        'episodes': 10, **{f: float(np.mean([v[f] for v in vals])) for f in vals[0]}}
                    rows.append(row); local.append(row)
                if len(local) == 3:
                    groups.append({'method': method, 'arm': arm, 'config': list(cfg),
                        **{f: stats([r[f] for r in local]) for f in metrics(eps[method, 0, cfg, SCENES[0], arm])}})
        for method, arms in (('regir_nomem', DEPTHS[:3]), ('refil', ('standard',))):
            for arm in arms:
                ps, scenes = [], []
                for seed in range(3):
                    keys = [(method, seed, cfg, scene, arm) for scene in SCENES]
                    controls = [('regir_nomem', seed, cfg, scene, 'L4') for scene in SCENES]
                    if not all(k in eps for k in keys + controls): continue
                    diffs = [metrics(eps[k])['R'] - metrics(eps[b])['R'] for k, b in zip(keys, controls)]
                    ps.append(float(np.mean(diffs)))
                    scenes.append({'training_seed': seed, 'delta_R_by_scene': diffs})
                if len(ps) == 3:
                    paired.append({'method': method, 'arm': arm, 'config': list(cfg),
                        'control': 'LEAF L4', 'delta_R': stats(ps), 'scene_pairs': scenes})
    data['summary'] = {'unique_cells': len(eps), 'reused_cells': sum(e['provenance']['kind'] == 'reused' for e in eps.values()),
        'new_cells': sum(e['provenance']['kind'] == 'new' for e in eps.values()), 'per_seed': rows,
        'across_seeds': groups, 'paired_reward': paired,
        'statistics': 'Ten scene means per training model; equal-weight mean and sample SD over three model seeds. Reward differences paired by model/configuration/scene.'}
    data['summary']['fresh_forward_attempts'] = data['summary']['new_cells'] + sum(
        a.get('additional_closed_loop_attempts', 0) for a in data.get('failed_attempts', []))
    data['execution_status'] = 'complete' if set(eps) == expected_keys() else 'partial'


def initialize(old, ids):
    data = {'schema': SCHEMA, 'episodes': [], 'checkpoints': [], 'training_updates': 0,
        'evaluation': {'configs': [list(c) for c in CONFIGS], 'training_seeds': [0, 1, 2],
            'episode_seeds': list(SCENES), 'LEAF_depths': [1, 2, 3, 4], 'REFIL': 'standard',
            'planned_unique_closed_loop_cells': 450, 'max_steps': 100, 'task_mode': 'damage',
            'spatial_dim': 2, 'target_initialization': 'random', 'blue_strategy': BLUE_STRATEGY,
            'checkpoint_kind': 'final', 'scope': 'Current-local frozen-final normal-depth evaluation; no replacement of formal training or evaluation records.',
            'policy_seed': 'SeedSequence([training_seed,N_R,N_B,K,episode_seed])',
            'L4_capture': 'Normal single forward only: all four read vectors, learned alpha, live IDs, pre-action physical state/actions; post-transition events saved at episode level.',
            'pressure': {'formula': 'sum_live_blue clip(1 - nearest_target_distance/4000,0,1) / initial_N_B',
                'distance_scale_m': 4000, 'source': 'had_wrapper._potential_by_target; -Phi/(shaping_coef * initial_N_B)',
                'target_set': 'all target positions, matching training potential; live attacker health>0',
                'interpretation': 'Factual attacker proximity pressure, not a fitted reward predictor.'}},
        'runtime': {'python': sys.version, 'python_executable': sys.executable, 'torch': torch.__version__,
            'numpy': np.__version__, 'device': 'cuda', 'gpu': torch.cuda.get_device_name(0), 'threads': 1,
            'tf32_matmul': False, 'tf32_cudnn': False, 'had_module': str(Path(had_env.__file__).resolve()),
            'had_core_version': CORE_VERSION, 'had_physics_protocol': PHYSICS_PROTOCOL},
        'source': {'runner': 'Papers/v3/analysis/mechanism_leaf1008.py', 'old_raw': str(OLD.relative_to(REPO)),
            'old_runtime': old['runtime'], 'old_analysis_scope': old.get('analysis_scope'),
            'formal_data_modified': False, 'source_files': ['Open-SCORE/open_score/models/entity_encoder.py',
                'Open-SCORE/open_score/envs/had_wrapper.py'], 'had_dependency': 'External PYTHONPATH source; no bytecode writes'},
        'execution_status': 'partial'}
    for e in old['episodes']:
        cfg = tuple(e['config']); method, arm = e['method'], e['arm']
        if cfg not in CONFIGS or e['training_seed'] not in range(3) or e['episode_seed'] not in SCENES: continue
        if not ((method == 'regir_nomem' and arm in ('L1', 'L2')) or (method == 'refil' and arm == 'standard')): continue
        if e['checkpoint_kind'] != 'final' or e['checkpoint_id'] != ids[method, e['training_seed']]:
            raise RuntimeError('Reused episode checkpoint mismatch')
        metrics(e)
        item = {k: e[k] for k in ('method', 'training_seed', 'config', 'episode_seed', 'arm',
            'checkpoint_kind', 'checkpoint_id', 'episode_summary', 'events', 'elapsed_seconds')}
        item['provenance'] = {'kind': 'reused', 'source': str(OLD.relative_to(REPO)),
            'original_arm': arm, 'original_updated_at': old.get('updated_at'),
            'required_fields': 'Matching final ID/configuration/model seed/scene seed; complete four outcome metrics. No full read-vector claim for reused cells.'}
        data['episodes'].append(item)
    summarize(data)
    if data['summary']['reused_cells'] != 180: raise RuntimeError('Unexpected matching reuse coverage')
    return data


def finite_stats(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return {'mean': float(x.mean()) if len(x) else None,
        'sample_sd': float(x.std(ddof=1)) if len(x) > 1 else None,
        'per_seed': list(values), 'n_models': int(len(x))}


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return None
    return float(np.corrcoef(x, y)[0, 1])


def feature_analysis(data):
    """Offline summaries of normal L4 factual vectors and physical transitions."""
    episodes = {ek(e): e for e in data['episodes']}
    seed_rows, group_rows, episode_rows = [], [], []
    audit = {'L4_episodes': 0, 'live_defender_decisions': 0, 'physical_decision_steps': 0,
        'nonfinite_tensor_rows': 0, 'active_id_shape_mismatches': 0,
        'new_initial_states_compared': 0, 'new_initial_state_mismatches': 0,
        'L4_cross_model_initial_states_compared': 0, 'L4_cross_model_initial_state_mismatches': 0,
        'old_local_L4_replays_compared': 0, 'old_local_L4_replays_matched': 0}
    for e in episodes.values():
        if e['provenance']['kind'] != 'new': continue
        leaf = episodes['regir_nomem', int(e['training_seed']), tuple(e['config']), int(e['episode_seed']), 'L4']
        if e['method'] == 'regir_nomem' and e['arm'] == 'L4':
            if e['training_seed']:
                reference = episodes['regir_nomem', 0, tuple(e['config']), int(e['episode_seed']), 'L4']
                audit['L4_cross_model_initial_states_compared'] += 1
                audit['L4_cross_model_initial_state_mismatches'] += e['initial_state'] != reference['initial_state']
            continue
        audit['new_initial_states_compared'] += 1
        audit['new_initial_state_mismatches'] += e['initial_state'] != leaf['initial_state']
    for cfg in CONFIGS:
        cfg_rows = []
        for seed in range(3):
            eps = [episodes['regir_nomem', seed, cfg, scene, 'L4'] for scene in SCENES]
            pressure_steps = np.asarray([d['pressure'] for e in eps for d in e['decisions']], dtype=float)
            cuts = np.quantile(pressure_steps, [1/3, 2/3])
            bins = [[] for _ in range(3)]
            alphas, read_norms, weighted_sizes, weighted_fractions, cosines = [], [], [], [], []
            spatial_sds, temporal_sds, temporal_changes, step_means = [], [], [], []
            event_arrays = {'next_step_target_damage': [], 'next_step_first_damage': [], 'next_step_blue_removal': []}
            event_vectors = {k: [[], []] for k in event_arrays}
            cfg_episode_rows = []
            for e in eps:
                ds = e['decisions']
                audit['L4_episodes'] += 1
                audit['physical_decision_steps'] += len(ds)
                replay = e.get('old_local_replay_validation')
                if replay is not None:
                    audit['old_local_L4_replays_compared'] += 1
                    audit['old_local_L4_replays_matched'] += bool(replay['matched'])
                ep_alpha, ep_weighted_fraction, temporal, local_temporal_changes = [], [], [], []
                firsts = set(t for t in e['episode_summary']['first_damage_step_by_target'] if t is not None)
                last_by_id = {}
                for index, d in enumerate(ds):
                    a, u = unpack(d['alpha']).astype(float), unpack(d['reads']).astype(float)
                    n = len(d['active_defender_ids'])
                    if a.shape != (n, 4) or u.shape != (n, 4, 64):
                        audit['active_id_shape_mismatches'] += 1
                        raise RuntimeError('Read tensor/live-ID dimensions disagree')
                    audit['live_defender_decisions'] += n
                    audit['nonfinite_tensor_rows'] += int(np.sum(~np.isfinite(a).all(1) | ~np.isfinite(u).all((1, 2))))
                    if not np.allclose(a.sum(1), 1, atol=1e-6): raise RuntimeError('Invalid saved alpha normalization')
                    norms = np.linalg.norm(u, axis=2)
                    sizes = a * norms
                    fractions = sizes / np.maximum(sizes.sum(1, keepdims=True), 1e-30)
                    normalized = u / np.maximum(norms[..., None], 1e-30)
                    adjacent = (normalized[:, 1:] * normalized[:, :-1]).sum(2)
                    alphas.append(a); read_norms.append(norms); weighted_sizes.append(sizes)
                    weighted_fractions.append(fractions); cosines.append(adjacent)
                    ep_alpha.append(a); ep_weighted_fraction.append(fractions)
                    spatial_sds.append(a.std(0)); temporal.append(a.mean(0)); step_means.append(a.mean(0))
                    for j, identity in enumerate(d['active_defender_ids']):
                        if identity in last_by_id:
                            local_temporal_changes.append(np.abs(a[j] - last_by_id[identity]))
                        last_by_id[identity] = a[j]
                    before = d['physical_state']
                    if index + 1 < len(ds):
                        after = ds[index+1]['physical_state']
                        if after['step'] != d['step']+1: raise RuntimeError('Nonconsecutive normal physical states')
                        next_damage = sum(t['cumulative_damage'] for t in after['targets']) - sum(t['cumulative_damage'] for t in before['targets'])
                        after_blue = sum(b['alive'] for b in after['blue'])
                    elif d['step'] + 1 == e['episode_summary']['ep_len']:
                        # An immediate terminal transition can use the exact totals.
                        next_damage = e['episode_summary']['D'] - sum(t['cumulative_damage'] for t in before['targets'])
                    else:
                        # After the last defender dies the environment can continue.
                        # Its entire unobserved tail must not be attributed to one step.
                        next_damage = None
                    next_blue_removal = any(v.get('kind') == 'agents_destroyed' and v.get('side') == 'Blue'
                        and v.get('step') == d['step'] + 1 and bool(v.get('agent_ids')) for v in e['events'])
                    flags = {'next_step_target_damage': next_damage > 1e-12 if next_damage is not None else None,
                        'next_step_first_damage': d['step'] + 1 in firsts,
                        'next_step_blue_removal': next_blue_removal}
                    group = int(np.searchsorted(cuts, d['pressure'], side='right'))
                    bins[group].append({'pressure': d['pressure'], 'alpha': a, 'weighted_size': sizes,
                        'weighted_fraction': fractions, 'flags': flags})
                    for key, flag in flags.items():
                        event_arrays[key].append(float(flag) if flag is not None else None)
                        if flag is not None: event_vectors[key][int(flag)].append(a)
                temporal_sds.append(np.asarray(temporal).std(0))
                if local_temporal_changes: temporal_changes.append(np.mean(local_temporal_changes, axis=0))
                ep_row = {'config': list(cfg), 'training_seed': seed, 'episode_seed': e['episode_seed'],
                    'valid_adjacent_receiver_step_pairs': len(local_temporal_changes),
                    'mean_alpha': np.concatenate(ep_alpha).mean(0).tolist(),
                    'mean_weighted_read_fraction': np.concatenate(ep_weighted_fraction).mean(0).tolist(),
                    'mean_pressure': float(np.mean([d['pressure'] for d in ds])),
                    'mean_temporal_alpha_sd': np.asarray(temporal).std(0).tolist(), **metrics(e)}
                episode_rows.append(ep_row); cfg_episode_rows.append(ep_row)
            pressure_groups = []
            for i, items in enumerate(bins):
                if not items:
                    pressure_groups.append({'group': i, 'physical_steps': 0, 'active_defender_decisions': 0})
                    continue
                pressure_groups.append({'group': i, 'physical_steps': len(items),
                    'active_defender_decisions': int(sum(len(v['alpha']) for v in items)),
                    'mean_pressure': float(np.mean([v['pressure'] for v in items])),
                    'mean_alpha': np.concatenate([v['alpha'] for v in items]).mean(0).tolist(),
                    'mean_weighted_read_size': np.concatenate([v['weighted_size'] for v in items]).mean(0).tolist(),
                    'mean_weighted_read_fraction': np.concatenate([v['weighted_fraction'] for v in items]).mean(0).tolist(),
                    **{f'{key}_fraction': float(np.mean([v['flags'][key] for v in items if v['flags'][key] is not None]))
                        for key in event_arrays}})
            a = np.concatenate(alphas)
            row = {'config': list(cfg), 'training_seed': seed, 'episodes': 10,
                'physical_steps': len(pressure_steps), 'active_defender_decisions': len(a),
                'valid_adjacent_receiver_step_pairs': sum(r['valid_adjacent_receiver_step_pairs'] for r in cfg_episode_rows),
                'pressure_tertile_thresholds': cuts.tolist(), 'pressure_groups': pressure_groups,
                'mean_alpha': a.mean(0).tolist(), 'mean_read_norms': np.concatenate(read_norms).mean(0).tolist(),
                'mean_weighted_read_size': np.concatenate(weighted_sizes).mean(0).tolist(),
                'mean_weighted_read_fraction': np.concatenate(weighted_fractions).mean(0).tolist(),
                'mean_adjacent_read_cosines': np.concatenate(cosines).mean(0).tolist(),
                'mean_within_step_defender_alpha_sd': np.mean(spatial_sds, axis=0).tolist(),
                'mean_within_episode_temporal_alpha_sd': np.mean(temporal_sds, axis=0).tolist(),
                'mean_same_defender_adjacent_step_alpha_abs_change': np.mean(temporal_changes, axis=0).tolist(),
                'pressure_alpha1_correlation': correlation(pressure_steps, np.asarray(step_means)[-len(pressure_steps):, 0]),
                'events': {key: {'physical_steps_with_event': int(sum(v for v in flags if v is not None)),
                    'total_physical_steps': len(flags), 'known_transition_steps': sum(v is not None for v in flags),
                    'mean_alpha_without_event': np.concatenate(event_vectors[key][0]).mean(0).tolist() if event_vectors[key][0] else None,
                    'mean_alpha_with_event': np.concatenate(event_vectors[key][1]).mean(0).tolist() if event_vectors[key][1] else None}
                    for key, flags in event_arrays.items()},
                'episode_outcome_groups': []}
            for damaged in (False, True):
                selected = [r for r in cfg_episode_rows if bool(r['target_damage_incidence']) == damaged]
                row['episode_outcome_groups'].append({'target_damaged': damaged, 'episodes': len(selected),
                    'mean_alpha': np.mean([r['mean_alpha'] for r in selected], axis=0).tolist() if selected else None,
                    'mean_pressure': float(np.mean([r['mean_pressure'] for r in selected])) if selected else None,
                    'mean_interception_fraction': float(np.mean([r['interception_fraction'] for r in selected])) if selected else None})
            seed_rows.append(row); cfg_rows.append(row)
        fields = ('mean_alpha', 'mean_read_norms', 'mean_weighted_read_size', 'mean_weighted_read_fraction',
            'mean_adjacent_read_cosines', 'mean_within_step_defender_alpha_sd',
            'mean_within_episode_temporal_alpha_sd', 'mean_same_defender_adjacent_step_alpha_abs_change')
        grouped = {'config': list(cfg), **{key: vector_stats([r[key] for r in cfg_rows]) for key in fields},
            'valid_adjacent_receiver_step_pairs_by_model': [r['valid_adjacent_receiver_step_pairs'] for r in cfg_rows],
            'pressure_alpha1_correlation': finite_stats([r['pressure_alpha1_correlation'] for r in cfg_rows]), 'pressure_groups': []}
        for i in range(3):
            items = [r['pressure_groups'][i] for r in cfg_rows]
            if any(not r['physical_steps'] for r in items):
                grouped['pressure_groups'].append({'group': i, 'n_models': sum(bool(r['physical_steps']) for r in items)})
                continue
            grouped['pressure_groups'].append({'group': i, 'n_models': 3,
                **{key: vector_stats([r[key] for r in items]) for key in
                    ('mean_alpha', 'mean_weighted_read_size', 'mean_weighted_read_fraction')},
                **{key: stats([r[key] for r in items]) for key in ('mean_pressure',
                    'next_step_target_damage_fraction', 'next_step_first_damage_fraction', 'next_step_blue_removal_fraction')}})
        group_rows.append(grouped)
    data['feature_analysis'] = {'integrity': audit, 'per_seed': seed_rows, 'across_seeds': group_rows,
        'per_episode': episode_rows,
        'definitions': {'pressure_grouping': 'Within configuration/model, tertiles of physical-decision-step pressure across ten scenes; np.searchsorted(side=right), thresholds/ties/counts saved. Group read means pool live receiver decisions within model; three model means receive equal weight.',
            'weighted_read_size': 'alpha_r * ||u_r||_2; descriptive vector magnitude, not marginal reward or signed output contribution.',
            'weighted_read_fraction': 'alpha_r ||u_r|| / sum_j alpha_j ||u_j||, calculated per live receiver before averaging; vector cancellation and learned projection excluded.',
            'event_time': 'pre-action state.step=t; next physical transition is t+1. Damage uses exact adjacent target cumulative-damage differences or an immediate terminal summary; unknown tails after all defenders die are excluded. First damage uses saved first_damage_step_by_target; blue removal uses agents_destroyed events at t+1.',
            'blue_removal': 'Decrease in live blue count on the next physical transition; not synonymous with interception. Exact interception is episode_summary.blue_deaths_by_cause.intercepted / initial N_B.',
            'defender_variation': 'Mean within-state population SD across live receiver alpha; temporal SD across state means within episode, then equal episode average.',
            'same_defender_adjacent_step_change': 'Within each episode, mean absolute alpha difference for the same surviving defender on consecutive physical steps, pooling valid receiver-step pairs; ten episode means receive equal weight within model, then three models receive equal weight. Not a temporal SD.',
            'event_association': 'Factual conditional description only; pressure/time/population composition and policy outcomes jointly vary. No attention/norm-to-reward causal claim.'}}


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
        'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42})
    folder = PAPER / 'figures'; folder.mkdir(parents=True, exist_ok=True)
    outcomes = {(tuple(g['config']), g['method'], g['arm']): g for g in data['summary']['across_seeds']}
    pairs = {(tuple(g['config']), g['method'], g['arm']): g for g in data['summary']['paired_reward']}
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.1))
    for column, cfg in enumerate(CONFIGS):
        ax, delta = axes[:, column]
        for j, arm in enumerate(DEPTHS):
            metric = outcomes[cfg, 'regir_nomem', arm]['R']
            ax.errorbar(j+1, metric['mean'], yerr=metric['sample_sd'], fmt='o', color='#17669b',
                capsize=3, markersize=4, linewidth=1.1, zorder=3)
            ax.scatter(j+1+np.array([-.09, 0, .09]), metric['per_seed'], color='#17669b', alpha=.45, s=11, zorder=4)
        reference = outcomes[cfg, 'refil', 'standard']['R']
        ax.axhline(reference['mean'], color='#b55a32', linewidth=1.2, linestyle='--')
        ax.fill_between([.65, 4.35], reference['mean']-reference['sample_sd'],
            reference['mean']+reference['sample_sd'], color='#b55a32', alpha=.09, linewidth=0)
        ax.set(title=f'({chr(97+column)}) {cfg[0]}v{cfg[1]}, K{cfg[2]}', xticks=[1, 2, 3, 4],
            xlim=(.65, 4.35), ylabel='Physical return R = -D' if column == 0 else None)
        for j, arm in enumerate(DEPTHS[:3]):
            metric = pairs[cfg, 'regir_nomem', arm]['delta_R']
            delta.errorbar(j+1, metric['mean'], yerr=metric['sample_sd'], fmt='o', color='#17669b',
                capsize=3, markersize=4, linewidth=1.1)
            delta.scatter(j+1+np.array([-.09, 0, .09]), metric['per_seed'], color='#17669b', alpha=.45, s=11, zorder=4)
        delta.axhline(0, color='#777777', linestyle='--', linewidth=.8)
        delta.set(title=f'({chr(100+column)}) Paired depth differences', xticks=[1, 2, 3], xlim=(.65, 3.35),
            xlabel='Normal execution depth L', ylabel='Paired R(L) - R(4)' if column == 0 else None)
        for panel in (ax, delta):
            panel.grid(axis='y', color='#dddddd', linewidth=.5); panel.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color='#17669b', marker='o', linestyle='', markersize=4, label='LEAF: mean ± model SD; small points = models'),
        Line2D([], [], color='#b55a32', linestyle='--', label='REFIL: mean ± model SD band')],
        loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1))
    fig.subplots_adjust(left=.085, right=.985, bottom=.115, top=.87, hspace=.5, wspace=.35)
    base = folder / 'fig_leaf_depth'
    fig.savefig(base.with_suffix('.pdf')); fig.savefig(base.with_suffix('.png'), dpi=300); plt.close(fig)

    feature = {tuple(g['config']): g for g in data['feature_analysis']['across_seeds']}
    colors = ('#17669b', '#d46a35', '#76539b', '#4c956c')
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    colors = ('#17669b', '#d46a35', '#76539b')
    labels = [f'{c[0]}v{c[1]}\nK{c[2]}' for c in CONFIGS]
    for cfg, color in zip(CONFIGS, colors):
        metric = feature[cfg]['mean_alpha']
        axes[0].errorbar([1, 2, 3, 4], 100*np.asarray(metric['mean']),
            yerr=100*np.asarray(metric['sample_sd']), fmt='o-', color=color, capsize=2,
            linewidth=1, markersize=3, label=f'{cfg[0]}v{cfg[1]}, K{cfg[2]}')
    axes[0].set(title='(a) Readout weights', xticks=[1, 2, 3, 4], xlim=(.8, 4.2), ylim=(-3, 100),
        xlabel='Relation round', ylabel='Fusion weight (%)')
    axes[0].legend(loc='upper right', frameon=False, handlelength=1.2, borderpad=.1, handletextpad=.3)
    for panel, field in ((axes[1], 'mean_within_step_defender_alpha_sd'),
            (axes[2], 'mean_same_defender_adjacent_step_alpha_abs_change')):
        for j, (cfg, color) in enumerate(zip(CONFIGS, colors)):
            metric = feature[cfg][field]
            panel.errorbar(j, 100*metric['mean'][0], yerr=100*metric['sample_sd'][0],
                fmt='D', color=color, capsize=3, linewidth=1.1, markersize=4, zorder=3)
            panel.scatter(j+np.array([-.1, 0, .1]), 100*np.asarray(metric['per_seed'])[:, 0],
                color=color, alpha=.55, s=13, zorder=4)
        panel.set(xticks=[0, 1, 2], xticklabels=labels, xlim=(-.3, 2.3), ylim=(0, None))
    axes[1].set(title='(b) Receiver variation', ylabel=r'Receiver $\alpha_1$ SD (pp)')
    axes[2].set(title='(c) Temporal variation', ylabel=r'Adjacent-step $|\Delta\alpha_1|$ (pp)')
    for panel in axes:
        panel.grid(axis='y', color='#dddddd', linewidth=.5); panel.set_axisbelow(True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.23, top=.85, wspace=.57)
    base = folder / 'fig_leaf_readout'
    fig.savefig(base.with_suffix('.pdf')); fig.savefig(base.with_suffix('.png'), dpi=300); plt.close(fig)
    data['figures'] = [f'figures/{name}.{ext}' for name in ('fig_leaf_depth', 'fig_leaf_readout') for ext in ('pdf', 'png')]
    data['figure_semantics'] = {'width_inches': 7.16, 'minimum_font_points': 8,
        'depth': 'Normal L1-L4 of the same frozen final model; depths sampled during training. Top absolute R and REFIL reference; bottom paired scene differences to normal L4. Equal three-model mean and sample SD, with model points.',
        'readout': 'Normal L4 factual trajectories, no perturbations. (a) live receiver-decision mean alpha within model, equal three-model mean and sample SD. (b) within-state population SD of alpha1 across live receivers, averaged over physical decision states within model. (c) mean absolute change in alpha1 for the same surviving receiver on adjacent physical steps; valid pairs pooled within episode, ten episodes equally averaged within model. In (b,c), diamonds and errorbars are three-model mean and sample SD, circles are the three model values. Pressure, weighted read sizes and other diagnostics remain in the raw data.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize-only', action='store_true')
    opts = parser.parse_args()
    if opts.summarize_only:
        data = json.loads(gzip.decompress(DATA.read_bytes()))
        summarize(data)
        if data['execution_status'] != 'complete': raise RuntimeError('All450 cells required for final figures')
        for ck in data['checkpoints']:
            saved = torch.load(REPO / ck['path'], map_location='cpu', weights_only=False)
            if saved['artifact_id'] != ck['checkpoint_id']: raise RuntimeError('Offline checkpoint identity changed')
            ck['saved_config'] = saved['config']
            ck['saved_progress'] = {key: value for key, value in saved['progress'].items()
                if key in ('t_env', 'episode_num', 'updates', 'status', 'measurement_revision', 'best_t_env', 'latest_validation_D')}
        feature_analysis(data)
        plot(data)
        save(data, DATA)
        print(json.dumps({'execution_status': data['execution_status'], 'summary': {key: data['summary'][key] for key in
            ('unique_cells', 'reused_cells', 'new_cells', 'fresh_forward_attempts')},
            'feature_integrity': data['feature_analysis']['integrity'], 'figures': data['figures']}, indent=2))
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(0); random.seed(0); torch.manual_seed(0)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    old = json.loads(gzip.decompress(OLD.read_bytes()))
    summary = json.loads((OUTPUT / 'summary.json').read_text(encoding='utf-8'))
    registered = {(c['method'], int(c['seed'])): c for c in summary['checkpoints']
        if c.get('env') == 'had' and c['method'] in ('regir_nomem', 'refil') and c['kind'] == 'final'}
    ids = {key: c['artifact_id'] for key, c in registered.items()}
    data = json.loads(gzip.decompress(DATA.read_bytes())) if DATA.exists() else initialize(old, ids)
    if data['schema'] != SCHEMA or data['evaluation']['configs'] != [list(c) for c in CONFIGS]:
        raise RuntimeError('Resume specification changed')
    existing = {ek(e) for e in data['episodes']}
    save(data, DATA)
    print(json.dumps({'start_cells': len(existing), 'reused': data['summary']['reused_cells'],
        'new_remaining': 450 - len(existing)}), flush=True)
    refs = {ek(e): e for e in old['episodes']}
    for method in ('regir_nomem', 'refil'):
        for seed in range(3):
            path = REPO / registered[method, seed]['archived_path']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved['artifact_id'] != ids[method, seed]: raise RuntimeError('Registered checkpoint mismatch')
            ck = {'method': method, 'training_seed': seed, 'checkpoint_id': ids[method, seed],
                'kind': 'final', 'path': str(path.relative_to(REPO))}
            if ck not in data['checkpoints']: data['checkpoints'].append(ck)
            policy = load_policy(method, path); policy.set_device('cuda')
            control = Control(policy, method)
            name = f'leaf_v3_normal_{method}_{seed}'
            register_end_to_end_policy('red', name, 'Frozen normal-depth evaluation', lambda _, p=control: p)
            for cfg in CONFIGS:
                for scene in SCENES:
                    for arm in DEPTHS if method == 'regir_nomem' else ('standard',):
                        key = (method, seed, cfg, scene, arm)
                        if key in existing: continue
                        control.depth = int(arm[1:]) if method == 'regir_nomem' else 4
                        torch.manual_seed(int(np.random.SeedSequence([seed, *cfg, scene]).generate_state(1)[0]))
                        started = time.perf_counter()
                        with torch.inference_mode():
                            result = run_episode(red=cfg[0], blue=cfg[1], targets=cfg[2], seed=scene,
                                red_strategy={'architecture': 'end_to_end', 'policy': name}, blue_strategy=BLUE_STRATEGY,
                                max_steps=100, task_mode='damage', spatial_dim=2, target_initialization='random',
                                diagnostics=True, record_events=True, retain_trajectory=False, record=False)
                        ep = {'method': method, 'training_seed': seed, 'config': list(cfg), 'episode_seed': scene,
                            'arm': arm, 'checkpoint_kind': 'final', 'checkpoint_id': ids[method, seed],
                            'episode_summary': dict(result['episode_summary']), 'events': result['events'],
                            'initial_state': control.initial, 'elapsed_seconds': time.perf_counter() - started,
                            'provenance': {'kind': 'new', 'source': 'current local source; normal forward only'}}
                        if method == 'regir_nomem' and arm == 'L4':
                            ep['decisions'] = control.decisions
                            old_key = (*key[:4], 'standard')
                            if old_key in refs:
                                ref = refs[old_key]
                                ep['old_local_replay_validation'] = replay_check(ep['episode_summary'],
                                    {**ref['episode_summary'], 'checkpoint_id': ref['checkpoint_id'],
                                        'recorded_at': old.get('updated_at')})
                        data['episodes'].append(ep); existing.add(key)
                        summarize(data); save(data, DATA)
                        print(json.dumps({'completed': len(existing), 'new': data['summary']['new_cells'],
                            'method': method, 'seed': seed, 'config': cfg, 'scene': scene, 'arm': arm,
                            'R': -ep['episode_summary']['D'], 'T': ep['episode_summary']['ep_len'],
                            'seconds': round(ep['elapsed_seconds'], 2),
                            'old_local_match': ep.get('old_local_replay_validation', {}).get('matched')}), flush=True)
            if control.handle: control.handle.remove()
    print(json.dumps({'status': data['execution_status'], 'unique_cells': len(existing)}), flush=True)


if __name__ == '__main__':
    main()
