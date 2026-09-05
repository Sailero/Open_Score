"""Time-budgeted, vector-collected candidate learning with preserved checkpoints."""
from __future__ import annotations

from collections import deque
import copy
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from torch.distributions import Categorical

from open_score.grouping.storage import (atomic_checkpoint, atomic_json, append_jsonl, fingerprint,
    provenance, random_state, restore_random_state, seed_everything, sha256, replace_file)
from .actions import candidate_actions, rule_action
from .config import curriculum_scales
from .environment import make_env
from .learning import ppo_update, double_q_update, imitation_update
from .policy import CandidateNetwork
from .rewards import potential, shaped_reward


def reconcile_logs(output, counters):
    """Archive an interrupted log suffix newer than the committed checkpoint.

    Otherwise resuming a saved model would append duplicate episode/step IDs
    from unsaved experience. Never silently discard a malformed interior row.
    """
    for name in ('episodes.jsonl', 'training.jsonl', 'validation_history.jsonl'):
        path = Path(output)/name
        if not path.exists():
            continue
        lines = path.read_bytes().splitlines(keepends=True)
        kept, removed = [], []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if index != len(lines)-1:
                    raise ValueError(f'Malformed interior log row: {path}:{index+1}') from None
                removed.append(line)
                continue
            committed = (row.get('physical_steps', row.get('steps', 0)) <= counters['physical_steps']
                         and row.get('updates', 0) <= counters['updates']
                         and row.get('episode', 0) <= counters['episodes'])
            (kept if committed else removed).append(line)
        if removed:
            archived = Path(output)/'resume_discarded'
            archived.mkdir(exist_ok=True)
            (archived/f'{name}.{time.time_ns()}.bin').write_bytes(b''.join(removed))
            temporary = path.with_suffix('.repair.tmp')
            temporary.write_bytes(b''.join(line.rstrip(b'\r\n')+b'\n' for line in kept))
            replace_file(temporary, path)


class Trainer:
    def __init__(self, config, output, *, seconds, steps=0, resume=False, checkpoint_every=25000):
        self.config, self.output = config, Path(output)
        if config['gamma'] != 1.0:
            raise ValueError('This finite-horizon success protocol requires gamma=1')
        self.output.mkdir(parents=True, exist_ok=True)
        self.seconds, self.step_limit = float(seconds), int(steps)
        self.checkpoint_every = int(checkpoint_every)
        self.evidence = provenance()
        seed_everything(config['seed'])
        self.network = CandidateNetwork(**config['model']).to(config['device'])
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config['learning_rate'])
        self.target = copy.deepcopy(self.network).eval().requires_grad_(False)
        self.counts = dict(physical_steps=0, upper_events=0, episodes=0, wins=0, updates=0,
                           optimizer_steps=0, next_episode=0, training_seconds=0.0,
                           teacher_simulation_steps=0, teacher_examples=0, teacher_seconds=0.0)
        self.replay = deque(maxlen=config['replay_capacity'])
        self.teacher = deque(maxlen=config['teacher_capacity'])
        self.recent_wins = deque(maxlen=100)
        self.best_score, self.last_validation_seconds = -math.inf, 0.0
        self.stop_requested, self.last_metrics = False, {}
        self.slots, self.rollout = [], []
        latest = self.output / 'latest.pt'
        if latest.exists():
            if not resume:
                raise FileExistsError('Use --resume or a new output directory')
            payload = torch.load(latest, map_location=config['device'], weights_only=False)
            if payload['config_hash'] != fingerprint(config):
                raise ValueError('Overnight resume configuration changed')
            if (payload['provenance']['source_hash'] != self.evidence['source_hash']
                    or payload['provenance']['assets'] != self.evidence['assets']):
                raise ValueError('Overnight source or frozen assets changed; preserve this output and use a new one')
            self.network.load_state_dict(payload['model'])
            self.optimizer.load_state_dict(payload['optimizer'])
            self.target.load_state_dict(payload['target'])
            self.counts.update(payload['counters'])
            self.replay.extend(payload.get('replay', []))
            self.teacher.extend(payload.get('teacher', []))
            self.recent_wins.extend(payload.get('recent_wins', []))
            self.best_score = payload.get('best_score', -math.inf)
            self.last_validation_seconds = payload.get('last_validation_seconds', 0.0)
            restore_random_state(payload['random_state'])
            reconcile_logs(self.output, self.counts)
        self.prior_seconds = self.counts['training_seconds']
        self.started = time.monotonic()
        self.last_saved = self.started
        self.last_saved_step = self.counts['physical_steps']
        self.last_display = 0.0
        self.phase = 'training'
        if not (self.output / 'initialized.pt').exists():
            self.save('initialized.pt', include_buffers=False)
        if not latest.exists():
            self.save()
        atomic_json(self.output / 'manifest.json', dict(config=config, provenance=self.evidence,
                    parameters=sum(p.numel() for p in self.network.parameters()),
                    protocol=f"Full physical world; frozen lower {config['executor_scope']} adapter; candidate upper actions",
                    resume_semantics='Completed learning updates retained; discard pending rollout and restart physical episodes'))

    def elapsed(self):
        return self.prior_seconds + time.monotonic()-self.started

    def fraction(self):
        fraction = self.elapsed()/self.seconds
        if self.step_limit:
            fraction = max(fraction, self.counts['physical_steps']/self.step_limit)
        return min(1.0, fraction)

    def finished(self):
        return (self.stop_requested or self.elapsed() >= self.seconds
                or self.step_limit > 0 and self.counts['physical_steps'] >= self.step_limit)

    def save(self, name='latest.pt', include_buffers=True):
        self.counts['training_seconds'] = self.elapsed()
        payload = dict(schema='overnight-candidate-v3', config=self.config,
            config_hash=fingerprint(self.config), provenance=self.evidence,
            model={k:v.detach().cpu() for k,v in self.network.state_dict().items()},
            target={k:v.detach().cpu() for k,v in self.target.state_dict().items()},
            optimizer=self.optimizer.state_dict(), counters=dict(self.counts),
            replay=list(self.replay) if include_buffers else [],
            teacher=list(self.teacher) if include_buffers else [], recent_wins=list(self.recent_wins),
            random_state=random_state(), best_score=self.best_score,
            last_validation_seconds=self.last_validation_seconds)
        atomic_checkpoint(self.output/name, payload)
        self.last_saved = time.monotonic()
        self.last_saved_step = self.counts['physical_steps']

    def progress(self, force=False):
        now = time.monotonic()
        if not force and now-self.last_display < 15:
            return
        row = dict(self.counts, phase=self.phase, training_seconds=self.elapsed(),
            remaining_training_seconds=max(0.0, self.seconds-self.elapsed()),
            recent_success_rate=float(np.mean(self.recent_wins)) if self.recent_wins else 0.0,
            average_steps_per_second=self.counts['physical_steps']/max(self.elapsed(),1e-6),
            metrics=self.last_metrics, device=self.config['device'])
        atomic_json(self.output/'progress.json', row)
        print(f"[{self.config['route']}] {self.phase}: steps={row['physical_steps']:,} "
              f"updates={row['updates']} success100={row['recent_success_rate']:.3f} "
              f"remaining={row['remaining_training_seconds']/3600:.2f}h", flush=True)
        self.last_display = now

    def new_slot(self):
        scales = curriculum_scales(self.config, self.fraction())
        episode = self.counts['next_episode']
        # Deterministic curriculum roster, independent of test/validation seeds.
        scale = scales[episode % len(scales)]
        seed = self.config['seed'] + 100000 + episode
        self.counts['next_episode'] += 1
        env = make_env(scale, opponent=self.config['opponent'],
                       executor_scope=self.config['executor_scope'],
                       max_steps=50, command_interval=5, device='cpu')
        state = env.reset(seed=seed)
        return dict(env=env, state=state, candidates=candidate_actions(state, self.config['candidate_limit']),
                    scale=scale, seed=seed, native_return=0.0)

    def record_transition(self, slot, next_state, reward, done, info):
        self.counts['physical_steps'] += info['delta']
        self.counts['upper_events'] += 1
        slot['native_return'] += reward
        if done:
            self.counts['episodes'] += 1
            self.counts['wins'] += int(info['success'])
            self.recent_wins.append(int(info['success']))
            append_jsonl(self.output/'episodes.jsonl', dict(episode=self.counts['episodes'],
                seed=slot['seed'], scale=slot['scale'], success=bool(info['success']),
                physical_steps=self.counts['physical_steps'], length=next_state.step,
                native_return=slot['native_return'], phase=self.phase))

    def record_update(self):
        self.counts['training_seconds'] = self.elapsed()
        row = dict(self.last_metrics,
                   optimizer_steps_this_update=self.last_metrics.get('optimizer_steps', 1),
                   **self.counts, phase=self.phase, elapsed_seconds=self.elapsed())
        append_jsonl(self.output/'training.jsonl', row)

    def teacher_stage(self):
        if self.config['route'] != 'ppo_teacher':
            return
        allotted = self.seconds*self.config['teacher_fraction']
        prior_teacher = self.counts['teacher_seconds']
        started = time.monotonic()
        self.phase = 'simulation_teacher'
        slot = self.new_slot()
        try:
            while not self.finished() and prior_teacher+time.monotonic()-started < allotted:
                # At most 15% of a requested smoke step budget is consumed by
                # teacher deployment, leaving room for actual PPO updates.
                if self.step_limit and self.counts['physical_steps'] >= self.step_limit*.15:
                    break
                env, state, pools = slot['env'], slot['state'], slot['candidates']
                snapshot = env.snapshot()
                k = min(len(pools), self.config['teacher_candidates'])
                indices = [0, *np.random.choice(np.arange(1,len(pools)), k-1, replace=False).tolist()]
                scores = []
                branch_seed = self.config['seed'] + 9000000 + self.counts['upper_events']
                for index in indices:
                    env.restore(snapshot)
                    env.set_rng(branch_seed)
                    following, native, done, info = env.step(pools[index])
                    steps, score = info['delta'], native
                    while not done and steps < self.config['teacher_horizon']:
                        following, native, done, info = env.step(rule_action(following))
                        steps += info['delta']
                        score += native
                    self.counts['teacher_simulation_steps'] += steps
                    # Limited-horizon heuristic teacher, not an oracle optimal
                    # action. Use independent branch RNG, never future real RNG.
                    scores.append(score + potential(following, terminal=done))
                env.restore(snapshot)
                tied = np.flatnonzero(np.isclose(scores, max(scores), atol=1e-9, rtol=0))
                best = indices[int(np.random.choice(tied))]
                if max(scores)-min(scores) > 1e-4:
                    self.teacher.append(dict(state=state, candidates=pools, label=best))
                    self.counts['teacher_examples'] += 1
                    if len(self.teacher) >= min(8, self.config['batch_size']):
                        rows = [self.teacher[i] for i in np.random.choice(len(self.teacher), min(len(self.teacher), self.config['batch_size']), replace=False)]
                        loss = imitation_update(self.network, self.optimizer, rows, self.config)
                        self.counts['updates'] += 1
                        self.counts['optimizer_steps'] += 1
                        self.last_metrics = dict(teacher_loss=loss, teacher_examples=len(self.teacher))
                        if self.counts['updates'] % 10 == 0:
                            self.record_update()
                following, native, done, info = env.step(pools[best])
                self.record_transition(slot, following, native, done, info)
                if done:
                    env.close()
                    slot = self.new_slot()
                else:
                    slot.update(state=following, candidates=candidate_actions(following, self.config['candidate_limit']))
                self.counts['teacher_seconds'] = prior_teacher+time.monotonic()-started
                self.periodic()
        finally:
            slot['env'].close()
            self.counts['teacher_seconds'] = prior_teacher+time.monotonic()-started
            self.phase = 'training'
            if sys.exc_info()[0] is None:
                self.target.load_state_dict(self.network.state_dict())
                self.save()

    def update_ppo(self):
        if not self.rollout:
            return
        self.last_metrics = ppo_update(self.network, self.optimizer, self.rollout, self.config,
                                       self.fraction(), list(self.teacher))
        self.counts['updates'] += 1
        self.counts['optimizer_steps'] += self.last_metrics['optimizer_steps']
        self.rollout.clear()
        self.record_update()

    def periodic(self):
        self.progress()
        if (time.monotonic()-self.last_saved >= self.config['checkpoint_seconds']
                or self.counts['physical_steps']-self.last_saved_step >= self.checkpoint_every):
            # Never serialize an on-policy rollout under different weights.
            if self.config['route'] != 'candidate_q':
                self.update_ppo()
            self.save()

    def validate(self):
        from .evaluation import evaluate_models
        self.update_ppo() if self.config['route'] != 'candidate_q' else None
        self.last_validation_seconds = self.elapsed()
        self.save()
        previous_rng = random_state()
        self.phase = 'validation'
        try:
            result = evaluate_models(self.config, {'current': self.output/'latest.pt'},
                self.output/'validation'/f"step_{self.counts['physical_steps']:09d}_{sha256(self.output/'latest.pt')[:10]}",
                episodes=self.config['validation_episodes'], wall_seconds=120 if self.config['smoke'] else 300,
                validation=True)
            if result['complete']:
                score = float(np.mean([row['success_rate'] for row in result['table']]))
                if score > self.best_score:
                    self.best_score = score
                    self.save('best.pt', include_buffers=False)
                    atomic_json(self.output/'best_validation.json', dict(score=score,
                                steps=self.counts['physical_steps'], evaluation=result))
            append_jsonl(self.output/'validation_history.jsonl', dict(steps=self.counts['physical_steps'],
                           training_seconds=self.elapsed(), summary=result))
        finally:
            restore_random_state(previous_rng)
            self.phase = 'training'

    def run(self):
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop_requested', True))
        try:
            self.teacher_stage()
            self.slots = [self.new_slot() for _ in range(self.config['environments'])]
            while not self.finished():
                states, pools = [s['state'] for s in self.slots], [s['candidates'] for s in self.slots]
                with torch.no_grad():
                    scores, values = self.network(states, pools)
                    if self.config['route'] == 'candidate_q':
                        actions = scores.argmax(1).cpu().tolist()
                        epsilon = max(self.config['epsilon_end'], 1-self.fraction()/.6)
                        for i, pool in enumerate(pools):
                            if np.random.random() < epsilon:
                                actions[i] = int(np.random.randint(len(pool)))
                        logps = [0.0]*len(actions)
                    else:
                        distribution = Categorical(logits=scores)
                        sampled = distribution.sample()
                        actions, logps = sampled.cpu().tolist(), distribution.log_prob(sampled).cpu().tolist()
                    values = values.cpu().tolist()
                for i, slot in enumerate(self.slots):
                    next_state, native, done, info = slot['env'].step(pools[i][actions[i]])
                    next_pool = candidate_actions(next_state, self.config['candidate_limit'])
                    row = dict(state=states[i], candidates=pools[i], action=actions[i], env=i,
                        log_prob=logps[i], value=values[i], reward=shaped_reward(states[i], next_state, native, done),
                        native_reward=native, done=done, delta=info['delta'], next_state=next_state, next_candidates=next_pool)
                    (self.replay if self.config['route']=='candidate_q' else self.rollout).append(row)
                    self.record_transition(slot, next_state, native, done, info)
                    if done:
                        slot['env'].close()
                        self.slots[i] = self.new_slot()
                    else:
                        slot.update(state=next_state, candidates=next_pool)
                if self.config['route']=='candidate_q':
                    if len(self.replay) >= self.config['replay_warmup']:
                        sample = [self.replay[i] for i in np.random.choice(len(self.replay), min(self.config['batch_size'],len(self.replay)), replace=False)]
                        self.last_metrics = double_q_update(self.network, self.target, self.optimizer, sample, self.config)
                        self.last_metrics['epsilon'] = epsilon
                        self.counts['updates'] += 1
                        self.counts['optimizer_steps'] += 1
                        if self.counts['updates'] % self.config['q_target_every'] == 0:
                            self.target.load_state_dict(self.network.state_dict())
                        if self.counts['updates'] % 10 == 0:
                            self.record_update()
                elif len(self.rollout) >= self.config['rollout_events']:
                    self.update_ppo()
                self.periodic()
                if self.elapsed()-self.last_validation_seconds >= self.config['validation_interval'] and not self.finished():
                    self.validate()
            if self.config['route'] != 'candidate_q':
                self.update_ppo()
            if not self.stop_requested:
                self.validate()
        finally:
            # Unexpected errors must never replace the last valid checkpoint
            # with a partly failed/nonfinite learning update.
            if sys.exc_info()[0] is None:
                self.save()
            for slot in self.slots:
                slot['env'].close()
            signal.signal(signal.SIGINT, old_handler)
        self.progress(force=True)
        return dict(self.counts, stopped=self.stop_requested,
                    best_validation_score=self.best_score if math.isfinite(self.best_score) else None,
                    parameters=sum(p.numel() for p in self.network.parameters()))
