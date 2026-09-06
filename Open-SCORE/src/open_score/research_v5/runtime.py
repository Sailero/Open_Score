"""Persisted task context, reproducible collection, and shared GPU admission."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from open_score.grouping.storage import atomic_checkpoint, replace_file
from open_score.research_v4.runner import atomic_json, read_json, pid_alive
from open_score.research_v4.evaluation import read_records
from .protocol import CELLS, EpisodeSpec, digest, episode_spec, stable_seed
from .simulator import choose, make_env


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(x) for x in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'to_dict'):
        return serializable(value.to_dict())
    return value


def append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(serializable(row), ensure_ascii=False, allow_nan=False)+'\n')


class TaskContext:
    def __init__(self, run_dir, task_id, config=None):
        self.run_dir = Path(run_dir).resolve()
        self.config = config or read_json(self.run_dir/'shared/config_resolved.json')
        self.task_id = str(task_id)
        self.output = self.run_dir/self.task_id
        self.output.mkdir(parents=True, exist_ok=True)
        self.root = self.output
        self.seed = int(self.config['seed'])
        requested = self.config.get('device', 'auto')
        self.device = 'cuda' if requested == 'auto' and torch.cuda.is_available() else ('cpu' if requested == 'auto' else requested)
        self.smoke = bool(self.config.get('smoke'))
        self.cpu_quota = int(self.config.get('cpu_quotas', {}).get(self.task_id, 1))
        self.started = time.monotonic()
        self._progress = read_json(self.output/'progress.json', {})
        self._gpu_wait_s = float(self._progress.get('gpu_wait_s', 0.))
        self._streams = {}
        self._rate_key = None
        self._rate_start = None
        self.identity = read_json(self.run_dir/'shared/protocol_resolved.json', {})
        atomic_json(self.output/'manifest.json', dict(task_id=self.task_id, training_seed=self.seed,
                    protocol_hash=self.identity.get('protocol_hash'), config=self.config))

    def log(self, stream, row):
        metadata = dict(schema_version='v5.1', run_id=self.run_dir.name, task_id=self.task_id,
                        training_seed=self.seed, timestamp_utc=datetime.now(timezone.utc).isoformat())
        append(self.output/(stream if str(stream).endswith('.jsonl') else f'{stream}.jsonl'), {**metadata, **row})

    def progress(self, phase=None, **metrics):
        if isinstance(phase, dict):
            metrics = {**phase, **metrics}
            phase = metrics.pop('phase', None)
        incoming = serializable(metrics)
        for target, sources in {'completed_states':('states','labeled_states'),
                'total_states':('states_target',),'completed_episodes':('construction_episode',),
                'total_episodes':('episodes_target',),'epochs':('epochs_target',),
                'simulation_physical_steps':('offline_sim_steps','teacher_physical_steps'),
                'rolling_success_rate':('recent_success',)}.items():
            if target not in incoming:
                for source in sources:
                    if source in incoming:
                        incoming[target]=incoming[source]; break
        if 'actor_updates' in incoming:
            incoming['optimizer_steps']=incoming['actor_updates']+incoming.get('critic_updates',0)
        key=(phase,incoming.get('method_id',incoming.get('method')),
             incoming.get('evaluation_split'),incoming.get('checkpoint_id'))
        if key != self._rate_key:
            if self._progress.get('evaluation_split') == 'validation' and self._progress.get('evaluation_success_rate') is not None:
                self._progress['last_validation_success_rate']=self._progress['evaluation_success_rate']
            self._progress={k:v for k,v in self._progress.items() if k in ('last_validation_success_rate','gpu_wait_s')}
            self._rate_key,self._rate_start=key,(time.monotonic(),dict(incoming))
            self._progress.pop('eta_seconds',None)
            self._progress.pop('throughput_per_second',None)
        if self._rate_start:
            started,baseline=self._rate_start
            duration=time.monotonic()-started
            for done,totals in [('physical_steps',('total_physical_steps','target_steps')),
                                ('completed_episodes',('total_episodes',)),('completed_states',('total_states',)),
                                ('completed',('total',)),('epoch',('epochs',))]:
                total=next((incoming[k] for k in totals if k in incoming),None)
                if done in incoming and total is not None and duration>1:
                    delta=incoming[done]-baseline.get(done,0)
                    if delta>0:
                        incoming['throughput_per_second']=delta/duration
                        incoming['eta_seconds']=max(0,total-incoming[done])/(delta/duration)
                    break
        self._progress.update(incoming)
        if phase is not None:
            self._progress['phase'] = phase
        self._progress.update(task_id=self.task_id, pid=os.getpid(),
                              updated=datetime.now(timezone.utc).isoformat(),
                              session_elapsed_s=time.monotonic()-self.started,
                              gpu_wait_s=self._gpu_wait_s)
        atomic_json(self.output/'progress.json', self._progress)

    @contextmanager
    def gpu(self):
        if self.device != 'cuda':
            yield
            return
        lock = self.run_dir/'shared/.gpu.lock'
        token = f'{os.getpid()}:{time.time_ns()}'
        started = time.monotonic()
        while True:
            try:
                with lock.open('x', encoding='utf-8') as stream:
                    json.dump(dict(pid=os.getpid(), token=token), stream)
                break
            except FileExistsError:
                try:
                    previous = read_json(lock, {})
                except (OSError, json.JSONDecodeError):
                    previous = {}
                if previous.get('pid') and not pid_alive(previous['pid']):
                    try:
                        lock.unlink()
                    except OSError:
                        pass
                elif not previous.get('pid'):
                    try:
                        if time.time()-lock.stat().st_mtime > 5.:
                            lock.unlink(missing_ok=True)
                    except OSError:
                        pass
                time.sleep(.05)
        self._gpu_wait_s += time.monotonic()-started
        try:
            yield
        finally:
            # On Windows another contender can briefly hold an open read
            # handle while checking the PID. Retry release without deleting a
            # subsequent owner's lock or failing a completed optimizer update.
            for attempt in range(300):
                if read_json(lock, {}).get('token') != token:
                    break
                try:
                    lock.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == 299:
                        raise
                    time.sleep(.01)

    def checkpoint(self, name, payload):
        path = self.output/name
        atomic_checkpoint(path, payload)
        return path

    def load_checkpoint(self, name):
        path = self.output/name
        return torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None

    def spec(self, index, split='train', namespace=None):
        return episode_spec(self.seed, index, split, namespace or self.task_id, self.config['cells'])

    def make_env(self, spec_or_index, split='train', namespace=None):
        spec = self.spec(spec_or_index, split, namespace) if isinstance(spec_or_index, (int, np.integer)) else spec_or_index
        return make_env(spec)

    def manifest(self, split='test'):
        filename = 'evaluation_manifest.json' if split == 'test' else 'validation_manifest.json'
        values = read_json(self.run_dir/'shared'/filename)
        return [EpisodeSpec.from_dict(x) for x in values['episodes']]

    def collect_states(self, count, split='train', policy=None, namespace=None):
        namespace = namespace or self.task_id
        key = digest([namespace, split, int(count)])[:16]
        directory = (self.run_dir/'shared' if namespace == 'shared' else self.output)/'datasets'/key
        directory.mkdir(parents=True, exist_ok=True)
        result = []
        index = 0
        while len(result) < count:
            path = directory/f'family_{index:06d}.pt'
            if path.exists():
                data = torch.load(path, map_location='cpu', weights_only=False)
            else:
                spec = self.spec(index, split, namespace)
                env = self.make_env(spec)
                if hasattr(policy, 'reset'):
                    policy.reset()
                trajectory, incoming, work, planning_work = [], 'initial', 0, 0
                try:
                    while not env.done:
                        state = env.state()
                        trajectory.append((state, env.snapshot(), incoming))
                        action = choose(policy, env)
                        trace = getattr(policy, 'last_trace', {}) or {}
                        planning_work += int(trace.get('simulation_physical_steps', trace.get('online_planner_steps',0)) or 0)
                        _, _, _, info = env.step(action)
                        work += info['delta']
                        incoming = info['event_reason']
                finally:
                    env.close()
                # One state per family keeps exact 15-cell balance even at odd quotas.
                category = ('initial', 'middle', 'casualty', 'threat')[(index//len(self.config['cells'])) % 4]
                candidates = list(range(len(trajectory)))
                if category == 'initial':
                    chosen = 0
                elif category == 'middle':
                    chosen = len(trajectory)//2
                elif category == 'casualty':
                    matches = [i for i, (_, _, event) in enumerate(trajectory) if event == 'casualty']
                    chosen = matches[0] if matches else len(trajectory)//2
                else:
                    def distance(i):
                        s = trajectory[i][0]
                        return min((np.linalg.norm(np.asarray(b.position)-np.asarray(t.position))
                                    for b in s.alive('blue') for t in s.alive('targets')), default=float('inf'))
                    chosen = min(candidates, key=distance)
                state, snapshot, actual_event = trajectory[chosen]
                row = dict(state=state, snapshot=snapshot, family_id=spec.family_id,
                           state_id=f'{spec.family_id}:event:{chosen}', episode_spec=spec.to_dict(),
                           state_kind=category, actual_event=actual_event,
                           category_available=category != 'casualty' or actual_event == 'casualty',
                           collection_physical_steps=work, collection_planning_steps=planning_work)
                data = dict(complete=True, row=row)
                atomic_checkpoint(path, data)
            result.append(data['row'])
            index += 1
            self.progress('collect_states', collection_split=split, completed_states=len(result), total_states=count)
        return result

    def bc_records(self):
        path = self.run_dir/'shared/bc_records.pt'
        data = torch.load(path, map_location='cpu', weights_only=False)
        return data['rows']

    def evaluate(self, policy, method_id, split='test', checkpoint='latest', limit=None, **kwargs):
        from .evaluate import evaluate_policy
        split = kwargs.pop('manifest_kind', split)
        started=time.monotonic()
        result=evaluate_policy(self, policy, method_id, split, checkpoint, limit)
        self.log('phase_costs',dict(profiled_by='context',phase='evaluation',method_id=method_id,
                 split=split,checkpoint=checkpoint,episodes=result['episodes'],wall_s=time.monotonic()-started))
        return result

    def diagnose(self, policy, method_id, **kwargs):
        from .diagnostics import diagnose_policy
        return diagnose_policy(self, policy, method_id, **kwargs)
