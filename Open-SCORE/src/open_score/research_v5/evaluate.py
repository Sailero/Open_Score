"""Paired, resumable native episodes; compute and learning scores stay separate."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
import gzip
import json
import itertools
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import time

import numpy as np

from open_score.research_v4.actions import rule_grouping, partition_key
from .reporting import wilson
from .protocol import EpisodeSpec, digest, episode_spec
from .runtime import append, serializable
from .simulator import choose, make_env, step_with_trace
from .storage import Store, method_dir

_POLICY = None


def _worker_init(encoded):
    global _POLICY
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    _POLICY = pickle.loads(encoded)


def _relation(grouping, members):
    relation = set()
    for group in grouping.groups:
        relation.update(itertools.combinations(sorted(set(group.members).intersection(members)), 2))
    return relation


def episode(policy, spec):
    if isinstance(spec, dict):
        spec = EpisodeSpec.from_dict(spec)
    env = make_env(spec)
    if hasattr(policy, 'reset'):
        policy.reset()
    timings, events, sizes, traces = [], [], [], []
    deployed_events = reserve_events = 0
    reserve_exposure = alive_exposure = simulated = target_changes = 0
    relation_changes = []
    first_destroyed = None
    last_info = {}
    try:
        while not env.done:
            state = env.state()
            previous = state.previous.prune(state.ids('red'))
            rule = rule_grouping(state)
            started = time.perf_counter()
            action = choose(policy, env)
            duration = time.perf_counter()-started
            action.validate(state.ids('red'), state.ids('targets'), max_members=None)
            timings.append(duration)
            trace = copy.deepcopy(getattr(policy, 'last_trace', {}) or {})
            live = state.ids('red')
            if live:
                deployed_events += int(bool(action.groups))
                reserve_events += int(not action.groups)
            before, after = previous.assignment(), action.assignment()
            changed = sum(before[i] != after[i] for i in live) if state.step else 0
            target_changes += changed
            pairs = len(live)*(len(live)-1)//2
            relation = len(_relation(previous, live) ^ _relation(action, live))/pairs if pairs and state.step else None
            if relation is not None:
                relation_changes.append(relation)
            unchanged = [i for i in live if before[i] == after[i]]
            unchanged_pairs = len(unchanged)*(len(unchanged)-1)//2
            same_relation = (len(_relation(previous, unchanged) ^ _relation(action, unchanged))/unchanged_pairs
                             if state.step and unchanged_pairs else None)
            sizes.extend(len(g.members) for g in action.groups)
            following, _, _, info = step_with_trace(env, action)
            last_info = info
            frames = info.pop('physical_trace')
            for frame in frames:
                reserve_exposure += frame['reserve_alive_before']
                alive_exposure += frame['red_alive_before']
                destroyed = [k for k, hp in frame['target_hp'].items() if hp <= 0]
                if first_destroyed is None and destroyed:
                    first_destroyed = (destroyed[0], frame['physical_step'])
            work = int(trace.get('simulation_physical_steps', trace.get('online_planner_steps', 0)) or 0)
            simulated += work
            event_id = f'{spec.family_id}:{state.step}'
            event = dict(event_id=event_id, family_id=spec.family_id, physical_step=state.step,
                remaining_steps=50-state.step, steps_until_periodic=5-state.step%5,
                event_type='initial' if state.step == 0 else events[-1]['next_event_type'],
                state_hash=digest(state.to_dict()), state=state.to_dict(),
                current_plan_raw=previous.to_dict(), rule_plan_raw=rule.to_dict(), selected_plan_raw=action.to_dict(),
                all_reserve_selected=bool(live and not action.groups), deployed_member_count=sum(map(lambda g:len(g.members), action.groups)),
                group_count=len(action.groups), target_change_count=changed, group_relation_change=relation,
                same_target_relation_change=same_relation, selection_wall_ms=duration*1000,
                physical_steps=info['delta'], next_event_type=info['event_reason'],
                rollout_physical_steps=work, decision_trace=trace, physical_trace=frames,
                actual_lower_actions=dict(env.executor.last_actions),
                target_hp={str(t.id):t.health for t in following.targets})
            events.append(event)
        count = deployed_events+reserve_events
        result = dict(**spec.to_dict(), evaluation_family_id=spec.family_id,
            n_red_initial=spec.red_count, n_blue_initial=spec.blue_count,
            success_native=bool(last_info.get('success')), success=bool(last_info.get('success')),
            termination_reason=last_info.get('event_reason'), physical_steps=env.state().step,
            command_events=len(events), target_hp_final={str(t.id):t.health for t in env.state().targets},
            target_first_destroyed=first_destroyed[0] if first_destroyed else None,
            first_target_destroy_step=first_destroyed[1] if first_destroyed else None,
            red_alive_final=len(env.state().ids('red')), blue_alive_final=len(env.state().ids('blue')),
            initial_all_reserve=events[0]['all_reserve_selected'] if events else None,
            never_deployed=bool(count and not deployed_events),
            all_reserve_event_fraction=reserve_events/count if count else None,
            reserve_exposure_fraction=reserve_exposure/alive_exposure if alive_exposure else None,
            group_size_mean=float(np.mean(sizes)) if sizes else 0.,
            nonempty_group_count_mean=float(np.mean([x['group_count'] for x in events])) if events else 0.,
            target_reassign_count=target_changes,
            pair_relation_change_mean=float(np.mean(relation_changes)) if relation_changes else None,
            decision_time_total_s=sum(timings), decision_time_p50_ms=float(np.quantile(timings,.5))*1000 if timings else None,
            decision_time_p95_ms=float(np.quantile(timings,.95))*1000 if timings else None,
            real_environment_steps=env.state().step, planner_sim_steps=simulated,
            execution_status='completed', invalid_action_count=0, repaired_action_count=0)
        return result, events
    finally:
        env.close()


def _work(spec):
    return episode(_POLICY, spec)


def summarize_rows(rows, cells, expected):
    groups = []
    for red, blue in cells:
        selected = [x for x in rows if x['red_count'] == red and x['blue_count'] == blue]
        wins = sum(x['success_native'] for x in selected)
        groups.append(dict(red_count=red, blue_count=blue, wins=wins, episodes=len(selected),
                           success_rate=wins/len(selected) if selected else None, wilson95=wilson(wins,len(selected))))
    wins = sum(x['success_native'] for x in rows)
    return dict(wins=wins, episodes=len(rows), success_rate=wins/len(rows) if rows else None,
                complete=len(rows)==expected, expected_episodes=expected, groups=groups,
                never_deployed_rate=float(np.mean([r['never_deployed'] for r in rows])) if rows else None)


def evaluate_policy(ctx, policy, method_id, split='test', checkpoint='latest', limit=None):
    directory = method_dir(ctx.run_dir, method_id)
    store = Store(directory)
    if split in ('test','validation'):
        specs = ctx.manifest(split)
    else:
        count = int(limit or ctx.config['validation_per_cell']*len(ctx.config['cells']))
        specs = [episode_spec(ctx.seed, i, split, 'shared', ctx.config['cells']) for i in range(count)]
    if limit is not None:
        specs = specs[:int(limit)]
    rows = store.episodes(method_id, split, checkpoint)
    done = {r['family_id'] for r in rows}
    if len(done) != len(rows):
        raise ValueError('Duplicate committed evaluation families')
    wanted = {s.family_id for s in specs}
    if not done <= wanted:
        raise ValueError('Evaluation manifest differs from committed episodes')
    pending = [s for s in specs if s.family_id not in done]
    metadata = dict(method_id=method_id, checkpoint_id=checkpoint, split=split, training_seed=ctx.seed,
                    protocol_hash=ctx.identity.get('protocol_hash'))
    def consume(result):
        row, events = result
        row.update(metadata)
        records=[dict(record_type='episode',**row)]
        for event in events:
            records.append(dict(record_type='event',**metadata, **event))
            decision = event.get('decision_trace', {})
            for i, candidate in enumerate(decision.get('candidates', decision.get('root_actions', []))):
                if not isinstance(candidate, dict):
                    continue
                clean = {k:v for k,v in candidate.items() if k not in ('state','branches')}
                records.append({'record_type':'candidate',**metadata, 'event_id':event['event_id'], 'candidate_id':i, **clean})
                for j, branch in enumerate(candidate.get('branches', [])):
                    if isinstance(branch, dict):
                        records.append({'record_type':'branch',**metadata, 'event_id':event['event_id'],
                            'candidate_id':i, 'branch_id':j, 'purpose':'selection', **branch})
            for j, draw in enumerate(decision.get('simulation_draws', [])):
                records.append(dict(record_type='branch',**metadata,event_id=event['event_id'], branch_id=j,
                                                    purpose='selection', **draw))
        data = ''.join(json.dumps(serializable(record),ensure_ascii=False,allow_nan=False)+'\n' for record in records)
        store.save_episode(method_id, split, checkpoint, row, gzip.compress(data.encode('utf-8')))
        rows.append(row)
        ctx.progress('evaluate', method_id=method_id, evaluation_split=split, checkpoint_id=checkpoint,
            completed_episodes=len(rows), total_episodes=len(specs),
            simulation_physical_steps=sum(x['planner_sim_steps'] for x in rows),
            evaluation_real_steps=sum(x['real_environment_steps'] for x in rows),
            deployment_rate=1-float(np.mean([x['never_deployed'] for x in rows])),
            evaluation_wins=sum(x['success_native'] for x in rows),
            evaluation_success_rate=sum(x['success_native'] for x in rows)/len(rows))
    if pending:
        encoded = pickle.dumps(policy)
        workers = min(ctx.cpu_quota, len(pending))
        if workers > 1:
            with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                                     initializer=_worker_init, initargs=(encoded,)) as pool:
                iterator = iter(pending)
                active = {pool.submit(_work, next(iterator).to_dict()) for _ in range(min(len(pending),workers*2))}
                while active:
                    ready, active = wait(active, return_when=FIRST_COMPLETED)
                    for future in ready:
                        consume(future.result())
                        spec = next(iterator, None)
                        if spec is not None:
                            active.add(pool.submit(_work, spec.to_dict()))
        else:
            for spec in pending:
                consume(episode(policy,spec))
    result = {**metadata, **summarize_rows(rows,ctx.config['cells'],len(specs)), 'path':str(directory)}
    store.put(f'evaluation/{method_id}/{split}/{checkpoint}', result)
    return result
