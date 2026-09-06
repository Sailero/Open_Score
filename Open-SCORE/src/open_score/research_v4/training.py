"""Recoverable three-route upper learning under a common pure rule executor."""
from __future__ import annotations

from collections import deque
import copy
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.distributions import Categorical

from open_score.grouping.storage import (atomic_checkpoint, atomic_json, append_jsonl,
    fingerprint, random_state, restore_random_state, seed_everything)
from open_score.grouping.domain import DecisionState, Grouping
from open_score.overnight.training import reconcile_logs
from .learning import CandidateNetwork, StateValueNetwork, ppo_update, double_q_update, imitation_update


ROUTES = ('r1_ppo', 'r2_teacher_ppo', 'r3_ddqn')


def configuration(config):
    """Normalize a JSON-compatible dict; budgets include teacher simulation."""
    supplied = dict(config)
    smoke = bool(supplied.get('smoke', False))
    defaults = dict(route='r1_ppo', seed=20260906, device='cpu', smoke=smoke,
        seconds=23400., steps=0, scales=[8, 12, 16, 24, 32], opponent='reactive',
        max_steps=50, command_interval=5, candidate_budget=32, envs=8,
        model=dict(hidden_dim=256, heads=8, layers=3), actor_learning_rate=5e-5,
        critic_learning_rate=2e-4, q_learning_rate=5e-5, rollout_events=1024,
        batch_size=128, epochs=4, gamma=1., gae_lambda=.95, clip=.2,
        target_kl=.03, entropy_start=.02, entropy_end=.005, max_gradient_norm=.5,
        replay_capacity=50000, replay_warmup=512, q_update_every=32, target_update_every=500,
        checkpoint_seconds=300., checkpoint_every=25000, validation_seconds=3600.,
        validation_episodes=100, validation_scales=[8, 12, 16, 24, 32],
        teacher_fraction=.12, teacher_candidates=6, teacher_branches=4,
        teacher_capacity=4096, teacher_updates_per_example=1, teacher_pretrain_updates=1000, torch_threads=1)
    if smoke:
        defaults.update(seconds=180., scales=[4, 8], envs=2, candidate_budget=6,
            model=dict(hidden_dim=32, heads=4, layers=1), rollout_events=16, batch_size=8,
            epochs=2, replay_capacity=128, replay_warmup=8, q_update_every=4,
            target_update_every=4, validation_episodes=2, validation_scales=[4, 8],
            validation_seconds=60., teacher_candidates=3, teacher_branches=2, teacher_pretrain_updates=8)
    if 'seconds' not in supplied and 'train_seconds' in supplied:
        supplied['seconds'] = supplied['train_seconds']
    if supplied.get('seconds') is None:
        supplied.pop('seconds', None)
    if 'threads' in supplied:
        supplied['torch_threads'] = supplied['threads']
    if 'candidate_limit' in supplied and 'candidate_budget' not in supplied:
        supplied['candidate_budget'] = supplied['candidate_limit']
    defaults.update(supplied)
    if defaults['route'] not in ROUTES or defaults['gamma'] != 1.:
        raise ValueError('Unknown route or non-native finite-horizon discount')
    for key in ('envs', 'candidate_budget', 'rollout_events', 'batch_size', 'epochs',
                'replay_capacity', 'replay_warmup', 'q_update_every', 'target_update_every',
                'validation_episodes', 'teacher_candidates', 'teacher_branches'):
        if int(defaults[key]) < 1:
            raise ValueError(f'{key} must be positive')
    if float(defaults['seconds']) <= 0 or int(defaults['steps']) < 0:
        raise ValueError('Positive time budget and nonnegative step limit required')
    if not defaults['scales'] or not defaults['validation_scales'] or not 0 <= defaults['teacher_fraction'] < 1:
        raise ValueError('Invalid scales or teacher fraction')
    if defaults['device'] == 'auto':
        defaults['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Paths accepted by callers are normalized before hashing/checkpointing.
    for key, value in list(defaults.items()):
        if isinstance(value, Path):
            defaults[key] = str(value.resolve())
    return defaults


def protocol_fingerprint():
    """Fingerprint source and physics without requiring any S1/S2 assets."""
    from .config import provenance
    return provenance()['source_hash']


def resume_config_hash(config):
    """Changing budgets is supported, changing the learned protocol is not."""
    return fingerprint({k:v for k, v in config.items() if k not in ('steps', 'seconds', 'train_seconds')})


class LearnedPolicy:
    def __init__(self, network, config):
        self.network, self.config = network.eval(), config

    @torch.no_grad()
    def act(self, state):
        from .actions import candidate_pool
        pools = candidate_pool(state, budget=self.config['candidate_budget'])
        scores, _ = self.network([state], [pools])
        return pools[int(scores[0].argmax())]

    __call__ = act


def load_policy(checkpoint, device='cpu'):
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if payload.get('schema') != 'research-v4-upper-1':
        raise ValueError('Not a v4 upper checkpoint')
    config = dict(payload['config'], device=str(device))
    actor = CandidateNetwork(**config['model']).to(device)
    actor.load_state_dict(payload['actor'])
    return LearnedPolicy(actor, config)


class Trainer:
    def __init__(self, config, output, resume=False):
        self.config, self.output = configuration(config), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.source_fingerprint = protocol_fingerprint()
        c = self.config
        torch.set_num_threads(int(c['torch_threads']))
        seed_everything(c['seed'])
        self.actor = CandidateNetwork(**c['model']).to(c['device'])
        self.critic = StateValueNetwork(**c['model']).to(c['device'])
        self.target = copy.deepcopy(self.actor).eval().requires_grad_(False)
        rate = c['q_learning_rate'] if c['route'] == 'r3_ddqn' else c['actor_learning_rate']
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c['critic_learning_rate'])
        self.counts = dict(physical_steps=0, learner_physical_steps=0, teacher_simulation_steps=0,
            validation_physical_steps=0, upper_events=0, episodes=0, wins=0, next_episode=0,
            updates=0, optimizer_steps=0, actor_updates=0, critic_updates=0,
            training_seconds=0., teacher_seconds=0., teacher_examples=0, teacher_labels=0,
            teacher_pretrain_updates=0, shared_teacher_simulation_steps=0)
        self.replay = deque(maxlen=c['replay_capacity'])
        self.teacher = deque(maxlen=c['teacher_capacity'])
        self.recent_wins = deque(maxlen=100)
        self.slots, self.rollout = [], []
        self.teacher_done = c['route'] != 'r2_teacher_ppo'
        self.teacher_data_loaded = False
        self.teacher_source = 'not_used' if self.teacher_done else 'online_terminal_simulation'
        self.annealing_budget = dict(seconds=c['seconds'], steps=c['steps'])
        self.best_wins = -1
        self.last_validation_seconds = 0.
        self.last_q_event = 0
        self.last_metrics = {}
        self.stop_requested = False
        self.phase = 'teacher' if not self.teacher_done else 'training'
        self.prior_seconds = 0.
        self.started = time.monotonic()
        latest = self.output/'latest.pt'
        if latest.exists():
            if not resume:
                raise FileExistsError('Output has a checkpoint; use resume or a new directory')
            self.restore(latest)
        elif resume and any(self.output.glob('*.pt')):
            raise ValueError('Cannot resume: latest.pt is missing')
        self.last_save_seconds = self.elapsed()
        self.last_save_steps = self.counts['physical_steps']
        self.last_progress = -math.inf
        if not latest.exists():
            self.save('initialized.pt')
            self.save()
        atomic_json(self.output/'manifest.json', dict(config=c, protocol_fingerprint=self.source_fingerprint,
            actor_parameters=sum(p.numel() for p in self.actor.parameters()),
            critic_parameters=sum(p.numel() for p in self.critic.parameters()),
            objective='Native global success only; no reward shaping; common pure rule executor',
            resume_semantics='All physical slots, pending rollout, replay, model, optimizers and RNG restored'))

    def elapsed(self):
        return self.prior_seconds+time.monotonic()-self.started

    def fraction(self):
        # Increasing a runtime cap continues optimization; it must not silently
        # reheat entropy or epsilon by moving their original decay denominator.
        budget = self.annealing_budget
        by_step = self.counts['physical_steps']/budget['steps'] if budget['steps'] else 0.
        return min(1., max(self.elapsed()/budget['seconds'], by_step))

    def finished(self):
        return (self.stop_requested or self.elapsed() >= self.config['seconds'] or
                self.config['steps'] > 0 and self.counts['physical_steps'] >= self.config['steps'])

    def _make_env(self, scale, seed):
        from .environment import make_env
        return make_env(scale, opponent=self.config['opponent'], seed=seed,
                        max_steps=self.config['max_steps'], command_interval=self.config['command_interval'])

    def new_slot(self):
        from .actions import candidate_pool
        index = self.counts['next_episode']
        scale = self.config['scales'][index % len(self.config['scales'])]
        seed = self.config['seed']+100000+index
        self.counts['next_episode'] += 1
        env = self._make_env(scale, seed)
        state = env.reset(seed=seed)
        return dict(env=env, state=state, candidates=candidate_pool(state, budget=self.config['candidate_budget']),
                    scale=scale, seed=seed, native_return=0.)

    def save(self, name='latest.pt'):
        self.counts['training_seconds'] = self.elapsed()
        slots = [dict(scale=s['scale'], seed=s['seed'], native_return=s['native_return'],
                      state=s['state'], candidates=s['candidates'], snapshot=s['env'].snapshot()) for s in self.slots]
        payload = dict(schema='research-v4-upper-1', config=self.config, config_hash=resume_config_hash(self.config),
            protocol_fingerprint=self.source_fingerprint,
            actor=self.actor.state_dict(), critic=self.critic.state_dict(), target=self.target.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            counters=dict(self.counts), random_state=random_state(), slots=slots, rollout=self.rollout,
            replay=list(self.replay), teacher=list(self.teacher), teacher_done=self.teacher_done,
            teacher_data_loaded=self.teacher_data_loaded, teacher_source=self.teacher_source,
            annealing_budget=self.annealing_budget,
            recent_wins=list(self.recent_wins), best_wins=self.best_wins,
            last_validation_seconds=self.last_validation_seconds, last_q_event=self.last_q_event)
        atomic_checkpoint(self.output/name, payload)
        self.last_save_seconds = self.elapsed()
        self.last_save_steps = self.counts['physical_steps']

    def restore(self, path):
        payload = torch.load(path, map_location=self.config['device'], weights_only=False)
        if payload.get('schema') != 'research-v4-upper-1' or payload['config_hash'] != resume_config_hash(self.config):
            raise ValueError('Resume configuration or checkpoint schema changed')
        old = payload['config']
        if (self.config['seconds'] < old['seconds'] or
                old['steps'] == 0 and self.config['steps'] != 0 or
                old['steps'] > 0 and 0 < self.config['steps'] < old['steps']):
            raise ValueError('Resume budgets may increase but must not decrease')
        if payload['protocol_fingerprint'] != self.source_fingerprint:
            raise ValueError('Resume source/physics protocol changed; use a new output')
        for name in ('actor', 'critic', 'target'):
            getattr(self, name).load_state_dict(payload[name])
        self.actor_optimizer.load_state_dict(payload['actor_optimizer'])
        self.critic_optimizer.load_state_dict(payload['critic_optimizer'])
        self.counts.update(payload['counters'])
        self.prior_seconds = self.counts['training_seconds']
        self.started = time.monotonic()
        self.replay.extend(payload['replay'])
        self.teacher.extend(payload['teacher'])
        self.rollout = payload['rollout']
        self.teacher_done = payload['teacher_done']
        self.teacher_data_loaded = payload.get('teacher_data_loaded', False)
        self.teacher_source = payload.get('teacher_source', self.teacher_source)
        self.annealing_budget = payload.get('annealing_budget', dict(seconds=old['seconds'], steps=old['steps']))
        self.recent_wins.extend(payload['recent_wins'])
        self.best_wins = payload['best_wins']
        self.last_validation_seconds = payload['last_validation_seconds']
        self.last_q_event = payload['last_q_event']
        for row in payload['slots']:
            env = self._make_env(row['scale'], row['seed'])
            restored = env.restore(row['snapshot'])
            if restored.to_dict() != row['state'].to_dict():
                raise ValueError('Restored physical state does not match committed observation')
            self.slots.append(dict(env=env, state=restored, candidates=row['candidates'],
                                  scale=row['scale'], seed=row['seed'], native_return=row['native_return']))
        restore_random_state(payload['random_state'])
        reconcile_logs(self.output, self.counts)

    def progress(self, force=False):
        if not force and time.monotonic()-self.last_progress < 15:
            return
        row = dict(self.counts, training_seconds=self.elapsed(), phase=self.phase,
            remaining_seconds=max(0., self.config['seconds']-self.elapsed()),
            recent_success_rate=float(np.mean(self.recent_wins)) if self.recent_wins else None,
            metrics=self.last_metrics)
        atomic_json(self.output/'progress.json', row)
        print(f"[{self.config['route']}] {self.phase}: learner={self.counts['learner_physical_steps']} "
              f"teacher={self.counts['teacher_simulation_steps']} updates={self.counts['updates']} "
              f"remaining={row['remaining_seconds']/60:.1f} min", flush=True)
        self.last_progress = time.monotonic()

    def record_update(self, metrics):
        self.counts['updates'] += 1
        self.counts['optimizer_steps'] += metrics.get('optimizer_steps', 0)
        self.counts['actor_updates'] += metrics.get('actor_updates', 0)
        self.counts['critic_updates'] += metrics.get('critic_updates', 0)
        self.last_metrics = metrics
        append_jsonl(self.output/'training.jsonl', dict(metrics, **self.counts,
            phase=self.phase, elapsed_seconds=self.elapsed(), optimizer_steps_this_update=metrics.get('optimizer_steps', 0)))

    def _advance(self, slot, action):
        from .actions import candidate_pool
        state, reward, done, info = slot['env'].step(action)
        delta = int(info['delta'])
        self.counts['physical_steps'] += delta
        self.counts['learner_physical_steps'] += delta
        self.counts['upper_events'] += 1
        slot['native_return'] += reward
        # A native terminal can have no surviving targets. Its value is masked
        # from Bellman/GAE targets, so do not generate nonexistent decisions.
        following = ([Grouping((), state.ids('red'))] if done else
                     candidate_pool(state, budget=self.config['candidate_budget']))
        if done:
            self.counts['episodes'] += 1
            self.counts['wins'] += int(info['success'])
            self.recent_wins.append(int(info['success']))
            append_jsonl(self.output/'episodes.jsonl', dict(episode=self.counts['episodes'],
                physical_steps=self.counts['physical_steps'], learner_physical_steps=self.counts['learner_physical_steps'],
                seed=slot['seed'], scale=slot['scale'], success=bool(info['success']),
                length=state.step, native_return=slot['native_return'], phase=self.phase))
        return state, reward, done, following

    def teacher_stage(self):
        """Paired full terminal ranking, followed by an irreversible phase switch."""
        if self.teacher_done:
            return
        from .data import paired_terminal
        from .actions import rule_grouping
        self.phase = 'teacher'
        shared = Path(self.config.get('shared_dir', ''))/'global_data'
        if shared.is_dir() or self.teacher_source == 'shared_terminal_data':
            self.shared_teacher_stage(shared)
            return
        if not self.slots:
            self.slots.append(self.new_slot())
        prior = self.counts['teacher_seconds']
        started = time.monotonic()
        allotted = self.config['seconds']*self.config['teacher_fraction']
        while not self.finished() and prior+time.monotonic()-started < allotted:
            # Sampling costs are not hidden outside a requested interaction cap.
            if self.config['steps'] and self.counts['physical_steps'] >= self.config['steps']*self.config['teacher_fraction']:
                break
            slot = self.slots[0]
            pools = slot['candidates']
            take = min(len(pools), self.config['teacher_candidates'])
            branches = self.config['teacher_branches']
            if self.config['steps']:
                # Keep enough real interaction for PPO after an expensive complete
                # teacher query, including very small smoke step requests.
                reserve = min(self.config['steps']//2,
                              self.config['rollout_events']*self.config['command_interval'])
                available = self.config['steps']-self.counts['physical_steps']-reserve
                horizon = max(1, self.config['max_steps']-slot['state'].step)
                take = min(take, available//horizon)
                if take < 2:
                    break
                branches = max(1, min(branches, available//(take*horizon)))
            indices = np.random.choice(len(pools), take, replace=False).tolist()
            pools = [pools[i] for i in indices]
            scores, steps = paired_terminal(slot['env'], slot['state'], pools,
                branches=branches, seed=self.config['seed']+9000000+self.counts['teacher_examples'],
                continuation=rule_grouping)
            self.counts['teacher_examples'] += 1
            self.counts['teacher_simulation_steps'] += int(steps)
            self.counts['physical_steps'] += int(steps)
            row = dict(state=slot['state'], candidates=pools, scores=scores)
            if np.ptp(scores) > 1e-12:
                self.teacher.append(row)
                self.counts['teacher_labels'] += 1
                for _ in range(self.config['teacher_updates_per_example']):
                    indices = np.random.choice(len(self.teacher), min(self.config['batch_size'], len(self.teacher)), replace=False)
                    metrics = imitation_update(self.actor, self.actor_optimizer,
                        [self.teacher[i] for i in indices], self.config['max_gradient_norm'])
                    self.record_update(metrics)
            # Visit the resulting state using the actual current policy; simulated
            # teacher futures never replace or leak into the live environment RNG.
            with torch.no_grad():
                logits, _ = self.actor([slot['state']], [slot['candidates']])
                index = int(Categorical(logits=logits).sample()[0])
            state, reward, done, next_pools = self._advance(slot, slot['candidates'][index])
            if done:
                slot['env'].close()
                self.slots[0] = self.new_slot()
            else:
                slot.update(state=state, candidates=next_pools)
            self.counts['teacher_seconds'] = prior+time.monotonic()-started
            self.maintenance()
        self.counts['teacher_seconds'] = prior+time.monotonic()-started
        if self.stop_requested:
            self.save()
            return
        self.teacher_done = True
        self.phase = 'training'
        # Teacher Adam moments must not silently become the PPO initialization.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config['actor_learning_rate'])
        self.save()

    def shared_teacher_stage(self, folder):
        """Use train families only; shared simulator work is accounted once.

        The source preparation already paid the simulation cost. Record that
        cost separately, never add it again to this learner's interaction cap.
        """
        from .data import read_dataset
        self.teacher_source = 'shared_terminal_data'
        if not self.teacher_data_loaded:
            source = read_dataset(folder, split='train')
            by_state = {}
            for row in source:
                if row.get('continuation_version') != 'rule_grouping_v1':
                    raise ValueError('R2 teacher data must use the frozen rule continuation')
                by_state.setdefault(row['state_id'], []).append(row)
            for rows in by_state.values():
                self.counts['teacher_examples'] += 1
                self.counts['shared_teacher_simulation_steps'] += sum(sum(r['physical_steps']) for r in rows)
                scores = [r['y'] for r in rows]
                if len(scores) > 1 and np.ptp(scores) > 1e-12:
                    self.teacher.append(dict(state=DecisionState.from_dict(rows[0]['state']),
                        candidates=[Grouping.from_dict(r['action']) for r in rows], scores=scores))
                    self.counts['teacher_labels'] += 1
            self.teacher_data_loaded = True
        prior, started = self.counts['teacher_seconds'], time.monotonic()
        allotted = self.config['seconds']*self.config['teacher_fraction']
        while (self.teacher and not self.finished() and prior+time.monotonic()-started < allotted
               and self.counts['teacher_pretrain_updates'] < self.config['teacher_pretrain_updates']):
            indices = np.random.choice(len(self.teacher), min(len(self.teacher), self.config['batch_size']), replace=False)
            metrics = imitation_update(self.actor, self.actor_optimizer, [self.teacher[i] for i in indices],
                                       self.config['max_gradient_norm'])
            self.counts['teacher_pretrain_updates'] += 1
            self.record_update(metrics)
            self.counts['teacher_seconds'] = prior+time.monotonic()-started
            self.maintenance()
        self.counts['teacher_seconds'] = prior+time.monotonic()-started
        if self.stop_requested:
            self.save()
            return
        self.teacher_done = True
        self.phase = 'training'
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config['actor_learning_rate'])
        self.save()

    def sample_replay(self):
        """Uniform scales followed by uniform transitions within the scale."""
        buckets = {}
        for row in self.replay:
            buckets.setdefault(row['scale'], []).append(row)
        scales = sorted(buckets)
        count = min(self.config['batch_size'], len(self.replay))
        return [buckets[scales[i % len(scales)]][np.random.randint(len(buckets[scales[i % len(scales)]]))]
                for i in range(count)]

    def training_stage(self):
        self.phase = 'training'
        c = self.config
        while len(self.slots) < c['envs'] and not self.finished():
            self.slots.append(self.new_slot())
        while not self.finished():
            with torch.no_grad():
                states = [s['state'] for s in self.slots]
                pools = [s['candidates'] for s in self.slots]
                scores, _ = self.actor(states, pools)
                if c['route'] == 'r3_ddqn':
                    epsilon = 1.-.95*min(1., self.fraction()/.8)
                    choices = [np.random.randint(len(p)) if np.random.random() < epsilon else int(scores[i].argmax())
                               for i, p in enumerate(pools)]
                    logp, values = [0.]*len(pools), [0.]*len(pools)
                else:
                    distribution = Categorical(logits=scores)
                    sampled = distribution.sample()
                    choices = sampled.cpu().tolist()
                    logp = distribution.log_prob(sampled).cpu().tolist()
                    _, value_tensor = self.critic(states, pools)
                    values = value_tensor.cpu().tolist()
            for i, slot in enumerate(self.slots):
                if self.finished():
                    break
                row = dict(state=slot['state'], candidates=slot['candidates'], action=choices[i],
                           log_prob=logp[i], value=values[i], env=i, scale=slot['scale'])
                state, reward, done, next_pools = self._advance(slot, slot['candidates'][choices[i]])
                row.update(reward=reward, done=done, next_state=state, next_candidates=next_pools)
                if c['route'] == 'r3_ddqn':
                    self.replay.append(row)
                else:
                    self.rollout.append(row)
                if done:
                    slot['env'].close()
                    self.slots[i] = self.new_slot()
                else:
                    slot.update(state=state, candidates=next_pools)
            if c['route'] == 'r3_ddqn':
                if len(self.replay) >= c['replay_warmup'] and self.counts['upper_events']-self.last_q_event >= c['q_update_every']:
                    metrics = double_q_update(self.actor, self.target, self.actor_optimizer, self.sample_replay(), c)
                    metrics['epsilon'] = epsilon
                    self.record_update(metrics)
                    self.last_q_event = self.counts['upper_events']
                    if self.counts['updates'] % c['target_update_every'] == 0:
                        self.target.load_state_dict(self.actor.state_dict())
            elif len(self.rollout) >= c['rollout_events']:
                self.record_update(ppo_update(self.actor, self.critic, self.actor_optimizer,
                    self.critic_optimizer, self.rollout, c, self.fraction()))
                self.rollout = []
            self.maintenance()
        # Native rollout truncation bootstraps. Never fabricate terminal rewards.
        if c['route'] != 'r3_ddqn' and self.rollout and not self.stop_requested:
            self.record_update(ppo_update(self.actor, self.critic, self.actor_optimizer,
                self.critic_optimizer, self.rollout, c, self.fraction()))
            self.rollout = []

    def validate(self):
        before_rng, before_phase = random_state(), self.phase
        self.phase = 'validation'
        rows = []
        started = time.monotonic()
        policy = LearnedPolicy(self.actor, self.config)
        try:
            for scale in self.config['validation_scales']:
                wins, episodes = 0, 0
                for index in range(self.config['validation_episodes']):
                    # Validation may consume the remaining train budget, but no
                    # held-out test data are ever consulted to select a model.
                    if self.stop_requested or self.elapsed() >= self.config['seconds']:
                        break
                    seed = self.config['seed']+20000000+scale*10000+index
                    env = self._make_env(scale, seed)
                    state, done = env.reset(seed=seed), False
                    try:
                        while not done:
                            state, _, done, info = env.step(policy.act(state))
                            self.counts['validation_physical_steps'] += int(info['delta'])
                        wins += int(info['success'])
                        episodes += 1
                    finally:
                        env.close()
                rows.append(dict(scale=scale, wins=wins, episodes=episodes,
                                 success_rate=wins/episodes if episodes else None))
            complete = all(row['episodes'] == self.config['validation_episodes'] for row in rows)
            wins = sum(row['wins'] for row in rows)
            record = dict(physical_steps=self.counts['physical_steps'], updates=self.counts['updates'],
                training_seconds=self.elapsed(), validation_seconds=time.monotonic()-started,
                wins=wins, episodes=sum(row['episodes'] for row in rows), complete=complete, scales=rows)
            append_jsonl(self.output/'validation_history.jsonl', record)
            self.last_validation_seconds = self.elapsed()
            if complete and wins > self.best_wins:
                self.best_wins = wins
                # Save after restoring the training random stream below.
                improved = True
            else:
                improved = False
        finally:
            self.actor.train()
            self.phase = before_phase
            restore_random_state(before_rng)
        if improved:
            self.save('best.pt')
        return record

    def maintenance(self):
        if self.teacher_done and self.elapsed()-self.last_validation_seconds >= self.config['validation_seconds'] and not self.finished():
            self.validate()
        if (self.elapsed()-self.last_save_seconds >= self.config['checkpoint_seconds'] or
                self.counts['physical_steps']-self.last_save_steps >= self.config['checkpoint_every']):
            self.save()
        self.progress()

    def run(self):
        result_path = self.output/'training_result.json'
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding='utf-8'))
            if prior.get('complete') and self.finished():
                return prior
        previous_handlers = {}
        def stop(signum, frame):
            self.stop_requested = True
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                try:
                    previous_handlers[signum] = signal.signal(signum, stop)
                except ValueError:
                    pass  # Unit tests may invoke the trainer from a non-main thread.
            self.teacher_stage()
            if not self.finished():
                self.training_stage()
            if self.elapsed() < self.config['seconds'] and not self.stop_requested:
                self.validate()
            if not (self.output/'best.pt').exists():
                # Explicitly tagged fallback; never describe it as validation-selected.
                self.save('best.pt')
            self.save()
            self.phase = 'stopped' if self.stop_requested else 'complete'
            self.progress(force=True)
            result = dict(route=self.config['route'], seed=self.config['seed'], complete=not self.stop_requested,
                counters=dict(self.counts), training_seconds=self.elapsed(),
                latest_checkpoint=str((self.output/'latest.pt').resolve()),
                best_checkpoint=str((self.output/'best.pt').resolve()),
                initialized_checkpoint=str((self.output/'initialized.pt').resolve()),
                teacher_source=self.teacher_source,
                annealing_budget=self.annealing_budget,
                best_validation_wins=self.best_wins,
                best_selection='validation' if self.best_wins >= 0 else 'latest_fallback_no_complete_validation',
                protocol_fingerprint=self.source_fingerprint)
            atomic_json(result_path, result)
            return result
        finally:
            # Unexpected failures preserve the last good checkpoint; contaminated
            # partial optimizer state must never overwrite it during cleanup.
            for slot in self.slots:
                slot['env'].close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def train(config, output, resume=False):
    return Trainer(config, output, resume=resume).run()
