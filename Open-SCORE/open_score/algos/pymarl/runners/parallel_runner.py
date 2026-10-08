"""ALMA parallel episode runner adapted for explicit jobs and Windows recovery.

Retains ALMA's EpisodeBatch, central batched MAC inference, complete episode
sampling and physical-step accounting. Evaluation reuses these same workers.
"""
from __future__ import annotations
from functools import partial
from multiprocessing import get_context
import traceback
import numpy as np


def _read_env(env):
    masks = env.get_masks()
    data = dict(masks)
    data['entities'] = np.asarray(env.get_entities(), dtype=np.float32)
    if hasattr(env, 'get_observer_entities'):
        data['observer_entities'] = np.asarray(env.get_observer_entities(), dtype=np.float32)
    data['avail_actions'] = np.asarray(env.get_avail_actions(), dtype=np.int32)
    na = len(data['avail_actions'])
    data['agent_mask'] = (1 - np.asarray(data['entity_mask'][:na])).astype(np.uint8)
    if hasattr(env, 'get_initial_agent_mask'):
        data['initial_agent_mask'] = env.get_initial_agent_mask()
    else:
        data['initial_agent_mask'] = np.zeros(na, dtype=np.uint8)
    data['state'] = np.asarray(env.get_state(), dtype=np.float32).reshape(-1)
    if hasattr(env, 'get_task_masks'):
        data.update(env.get_task_masks())
    return data


def env_worker(remote, args_dict, rank):
    """No policy, GPU context, snapshots, or training in an environment worker."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = None
    try:
        from open_score.algos import make_runtime_env
        env = make_runtime_env(args_dict, rank=rank)
        while True:
            cmd, payload = remote.recv()
            if cmd == 'close':
                env.close()
                remote.send(None)
                return
            if cmd == 'get_env_info':
                remote.send(env.get_env_info())
            elif cmd == 'reset':
                env.reset(**payload)
                remote.send(_read_env(env))
            elif cmd == 'step':
                reward, done, info = env.step(payload)
                result = _read_env(env)
                result.update(reward=float(reward), terminated=bool(done), info=info)
                if done:
                    result['summary'] = env.episode_summary() if hasattr(env, 'episode_summary') else {}
                    if hasattr(env, 'get_trajectory'):
                        result['trajectory'] = env.get_trajectory()
                remote.send(result)
            elif cmd == 'summary':
                remote.send(env.episode_summary() if hasattr(env, 'episode_summary') else {})
            elif cmd == 'get_rng_state':
                remote.send(env.get_rng_state() if hasattr(env, 'get_rng_state') else None)
            elif cmd == 'set_rng_state':
                if payload is not None and hasattr(env, 'set_rng_state'):
                    env.set_rng_state(payload)
                remote.send(None)
            else:
                raise ValueError(cmd)
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            remote.send({'_worker_error': traceback.format_exc()})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        remote.close()


class ParallelRunner:
    def __init__(self, args, logger):
        self.args, self.logger = args, logger
        self.batch_size = int(args.batch_size_run)
        self.parent_conns, self.ps = [], []
        context = get_context('spawn')
        for rank in range(self.batch_size):
            parent, child = context.Pipe()
            process = context.Process(target=env_worker, args=(child, vars(args), rank), daemon=True)
            process.start()
            child.close()
            self.parent_conns.append(parent)
            self.ps.append(process)
        self.parent_conns[0].send(('get_env_info', None))
        self.env_info = self._recv(0)
        self.episode_limit = int(self.env_info['episode_limit'])
        self.t_env = 0
        self.episode_counts = [0] * self.batch_size
        self.seed_rngs = [np.random.default_rng(np.random.SeedSequence([args.seed, 3107, i])) for i in range(self.batch_size)]
        self.last_episode_rows = []
        self.last_trajectories = []
        self.progress_callback = None

    def _recv(self, rank):
        while not self.parent_conns[rank].poll(1.0):
            if not self.ps[rank].is_alive():
                raise RuntimeError(f'Environment worker {rank} exited with code {self.ps[rank].exitcode}')
        data = self.parent_conns[rank].recv()
        if isinstance(data, dict) and '_worker_error' in data:
            raise RuntimeError(data['_worker_error'])
        return data

    def setup(self, scheme, groups, preprocess, mac):
        self.scheme, self.groups, self.preprocess, self.mac = scheme, groups, preprocess, mac

    def get_env_info(self):
        return self.env_info

    def _pre(self, values, t=None, terminal=None):
        observation_keys = ('entities', 'observer_entities', 'obs_mask', 'entity_mask', 'agent_mask',
                            'initial_agent_mask', 'state', 'avail_actions',
                            'entity2task_mask', 'task_mask')
        data = {key: np.stack([row[key] for row in values]) for key in observation_keys
                if key in self.scheme and key in values[0]}
        if 'hier_decision' in self.scheme:
            # Prefer the environment clock when it records one, so event-driven
            # ALMA and the official interval stay on the same runner path. A
            # terminal state is never a decision point.
            ended = list(terminal or [False] * len(values))
            if 'hier_decision' in values[0]:
                flags = []
                for row, done in zip(values, ended):
                    flag = int(np.asarray(row['hier_decision']).reshape(-1)[0])
                    flags.append([0 if done else flag])
                data['hier_decision'] = np.asarray(flags, dtype=np.uint8)
            else:
                length = int(self.args.hier_agent['action_length'])
                decide = int(t) % length == 0
                data['hier_decision'] = np.asarray(
                    [[int(decide and not done)] for done in ended], dtype=np.uint8)
        return data

    def run(self, test_mode=False, jobs=None, max_train_steps=None, **unused):
        # Windows imports the target module in each spawned environment
        # process. Keep Torch and replay imports local to the learner process.
        import torch
        with torch.no_grad():
            return self._collect(test_mode, jobs, max_train_steps, **unused)

    def _collect(self, test_mode=False, jobs=None, max_train_steps=None, **unused):
        from components.episode_buffer import EpisodeBatch
        if jobs is None:
            jobs = [{'episode_seed': int(rng.integers(100000, 2**31 - 1))} for rng in self.seed_rngs]
        if not 0 < len(jobs) <= self.batch_size:
            raise ValueError('One runner call accepts 1..batch_size_run explicit jobs')
        if max_train_steps is not None and not test_mode:
            jobs = jobs[:min(len(jobs), int(max_train_steps))]
        count = len(jobs)
        batch = EpisodeBatch(self.scheme, self.groups, count, self.episode_limit + 1,
                             preprocess=self.preprocess, device=self.args.device)
        self.batch = batch
        for rank, job in enumerate(jobs):
            payload = dict(seed=int(job['episode_seed']), evaluate=bool(test_mode))
            if job.get('config') is not None:
                payload['config'] = job['config']
            if job.get('reset_config') is not None:
                payload['reset_config'] = job['reset_config']
            if job.get('engine_seed') is not None:
                payload['engine_seed'] = int(job['engine_seed'])
            payload['retain_trajectory'] = bool(job.get('retain_trajectory', False))
            self.parent_conns[rank].send(('reset', payload))
        initial = [self._recv(rank) for rank in range(count)]
        batch.update(self._pre(initial, t=0), ts=0)
        if 't_added' in self.scheme:
            batch.update({'t_added': np.full((count, 1), self.t_env, dtype=np.int64)})
        self.mac.init_hidden(count)
        self.mac.eval()
        active = list(range(count))
        returns = [0.0] * count
        lengths = [0] * count
        summaries = [None] * count
        trajectories = [None] * count
        qstats = [dict(q_tot=[], q_i=[]) for _ in range(count)]
        physical_steps = 0
        t = 0
        while active:
            # At the declared budget boundary finish the collected trajectory
            # with a sampling reset, retaining bootstrap from its final state.
            if max_train_steps is not None and not test_mode:
                capacity = int(max_train_steps) - physical_steps
                waiting = active[max(0, capacity):]
                active = active[:max(0, capacity)]
                for rank in waiting:
                    self.parent_conns[rank].send(('summary', None))
                    summaries[rank] = dict(self._recv(rank))
                    # An environment summary already reports the physical
                    # return; 'returns' carries the shaped learning signal.
                    summaries[rank].update(shaped_return=returns[rank], ep_len=lengths[rank],
                                           episode_seed=int(jobs[rank]['episode_seed']))
                    summaries[rank].setdefault('return', returns[rank])
                    if t > 0:
                        batch.update({'reset': [[True]]}, bs=[rank], ts=t - 1, mark_filled=False)
                if not active:
                    break
            actions = self.mac.select_actions(batch, t_ep=t, t_env=self.t_env,
                                              bs=active, test_mode=test_mode)
            if test_mode and hasattr(self, 'value_callback'):
                values = self.value_callback(batch, t, actions, active)
                for i, rank in enumerate(active):
                    for key in ('q_tot', 'q_i'):
                        value = values[key][i]
                        if value is not None:
                            qstats[rank][key].append(float(value))
            batch.update({'actions': actions.unsqueeze(-1)}, bs=active, ts=t, mark_filled=False)
            cpu_actions = actions.cpu().numpy()
            for i, rank in enumerate(active):
                self.parent_conns[rank].send(('step', cpu_actions[i]))
            returned = [self._recv(rank) for rank in active]
            rewards, natural, resets = [], [], []
            subtask_rewards, subtask_terminated = [], []
            next_active = []
            for rank, result in zip(active, returned):
                info = result['info']
                done = result['terminated']
                is_natural = bool(info.get('terminal_for_learning',
                    info.get('terminated_naturally', done and not info.get('episode_limit', False))))
                returns[rank] += result['reward']
                # A folded Red-wipeout tail runs real physics inside one
                # environment step, so the budget must count those steps too.
                advanced = 1 + int(info.get('folded_steps', 0))
                lengths[rank] += advanced
                physical_steps += advanced
                rewards.append([result['reward']])
                natural.append([is_natural])
                resets.append([done])
                if 'task_rewards' in self.scheme:
                    subtask_rewards.append(info['task_rewards'])
                    subtask_terminated.append(info['tasks_terminated'])
                if done:
                    summaries[rank] = dict(result.get('summary') or {})
                    summaries[rank].setdefault('return', returns[rank])
                    summaries[rank].setdefault('ep_len', lengths[rank])
                    summaries[rank].setdefault('episode_seed', int(jobs[rank]['episode_seed']))
                    trajectories[rank] = result.get('trajectory')
                else:
                    next_active.append(rank)
            post = {'reward': rewards, 'terminated': natural, 'reset': resets}
            if subtask_rewards:
                post.update(task_rewards=subtask_rewards, tasks_terminated=subtask_terminated)
            batch.update(post, bs=active, ts=t, mark_filled=False)
            batch.update(self._pre(returned, t=t + 1, terminal=[row['terminated'] for row in returned]),
                         bs=active, ts=t + 1)
            active = next_active
            t += 1
            if self.progress_callback:
                self.progress_callback(test_mode, sum(s is not None for s in summaries), count, physical_steps)
        if not test_mode:
            self.t_env += physical_steps
            for rank in range(count):
                self.episode_counts[rank] += 1
        for rank, summary in enumerate(summaries):
            qs = qstats[rank]
            summary['q_tot_mean'] = float(np.mean(qs['q_tot'])) if qs['q_tot'] else None
            summary['q_tot_std'] = float(np.std(qs['q_tot'])) if qs['q_tot'] else None
            summary['q_i_mean'] = float(np.mean(qs['q_i'])) if qs['q_i'] else None
        self.last_episode_rows, self.last_trajectories = summaries, trajectories
        self.env_steps_this_run = physical_steps
        if getattr(self.args, 'leaf_loop_core', None):
            # Earlier-finishing environments have their own final time index.
            # Record bootstrap exits without a second action/GRU commit.
            for rank in range(count):
                filled = batch['filled'][rank, :, 0].nonzero(as_tuple=False).flatten()
                if filled.numel():
                    self.mac.prepare_loop_depth(batch, int(filled[-1]), bs=[rank], test_mode=test_mode)
        return batch[:, :t + 1], summaries

    def state_dict(self):
        for conn in self.parent_conns:
            conn.send(('get_rng_state', None))
        state = {'t_env': self.t_env, 'episode_counts': self.episode_counts,
                'seed_rngs': [rng.bit_generator.state for rng in self.seed_rngs],
                'env_rngs': [self._recv(i) for i in range(self.batch_size)]}
        if getattr(self.args, 'leaf_loop_core', None):
            state['loop_depth_rng'] = self.mac.loop_depth_rng_state()
        return state

    def load_state_dict(self, state):
        self.t_env = state['t_env']
        self.episode_counts = list(state['episode_counts'])
        if getattr(self.args, 'leaf_loop_core', None):
            self.mac.load_loop_depth_rng_state(state.get('loop_depth_rng'))
        for rng, saved in zip(self.seed_rngs, state['seed_rngs']):
            rng.bit_generator.state = saved
        for conn, saved in zip(self.parent_conns, state['env_rngs']):
            conn.send(('set_rng_state', saved))
        for i in range(self.batch_size):
            self._recv(i)

    def close_env(self):
        for conn, process in zip(self.parent_conns, self.ps):
            if process.is_alive():
                try:
                    conn.send(('close', None))
                except (EOFError, BrokenPipeError):
                    pass
        for i, process in enumerate(self.ps):
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            self.parent_conns[i].close()

    def save_replay(self):
        return None
