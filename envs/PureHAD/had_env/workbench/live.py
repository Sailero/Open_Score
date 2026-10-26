"""Owned spawn worker for a debug session; never attaches to training processes."""
from __future__ import annotations

import copy
import multiprocessing as mp
from queue import Empty
import time
import traceback


class _StopSession(Exception):
    pass


def _worker(commands, results, scenario, policy, native, save_path):
    try:
        _worker_session(commands, results, scenario, policy, native, save_path)
    except Exception:
        # Initialization errors also cross the process boundary.
        results.put({"kind": "error", "message": traceback.format_exc()})


def _worker_session(commands, results, scenario, policy, native, save_path):
    import os
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Built-in rules require no ML runtime; never load one solely for a worker.
    import sys
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)
    if native:
        from .agent_session import FlightSession
        session = FlightSession(scenario)
    else:
        from .protocols import ScenarioSpec
        from .session import SimulationSession
        session = SimulationSession(ScenarioSpec.from_dict(scenario), policy)
    state = dict(playing=False, permits=0, interval=1., last=0., branch=False,
                 stop_boundary=False, stop_event=False)

    def command(value):
        action = value.get("action") if isinstance(value, dict) else value
        if action == "stop":
            raise _StopSession()
        if action == "play":
            state["playing"] = True
        elif action == "pause":
            state.update(playing=False, permits=0, stop_boundary=False, stop_event=False)
        elif action == "step":
            state["playing"] = False
            state["permits"] += 1
        elif action == "speed":
            state["interval"] = 1. / max(.25, min(8., float(value["value"])))
        elif action == "next_decision":
            state.update(playing=True, stop_boundary=True)
        elif action == "next_event":
            state.update(playing=True, stop_event=True)
        elif action == "branch":
            state["branch"] = True
            results.put({"kind": "notice", "message": "分支将在下一个决策边界创建；中途暂停时请继续播放。"})

    def gate():
        while True:
            try:
                while True:
                    command(commands.get_nowait())
            except Empty:
                pass
            now = time.monotonic()
            if state["permits"] > 0:
                state["permits"] -= 1
                state["last"] = now
                return
            if state["playing"] and now - state["last"] >= state["interval"]:
                state["last"] = now
                return
            try:
                command(commands.get(timeout=.03))
            except Empty:
                pass

    target = session.env if native else session.env.adapter
    physical_step = target.step
    def gated(*args, **kwargs):
        gate()
        return physical_step(*args, **kwargs)
    target.step = gated
    original_append = session.recorder.append_frame
    def append(frame):
        original_append(frame)
        results.put({"kind": "frame", "frame": copy.deepcopy(frame)})
        casualty_kinds = {"death", "collision", "agents_destroyed", "target_destroyed", "targets_breached",
                          "self_destruct", "terminal", "red_win", "blue_win"}
        if state["stop_event"] and any(event.get("kind", event.get("type")) in casualty_kinds
                                      for event in frame.get("events", [])):
            state.update(playing=False, stop_event=False)
            results.put({"kind": "paused"})
    session.recorder.append_frame = append
    try:
        results.put({"kind": "episode", "episode": session.recorder.episode.to_dict()})
        while not session.done:
            session.step()
            results.put({"kind": "update", "episode": session.recorder.episode.to_dict()})
            if state["stop_boundary"]:
                state.update(playing=False, stop_boundary=False)
                results.put({"kind": "paused"})
            if state["branch"]:
                state["branch"] = False
                if session.done:
                    results.put({"kind": "notice", "message": "当前回合已结束，未创建分支；请从较早决策点分叉。"})
                    continue
                branch = session.branch(continuation_seed=20260909)
                try:
                    results.put({"kind": "branch", "episode": branch.run().to_dict()})
                finally:
                    branch.close()
        if save_path:
            session.recorder.save(save_path)
        results.put({"kind": "complete", "episode": session.recorder.episode.to_dict()})
    except _StopSession:
        if save_path:
            session.recorder.save(save_path)
        results.put({"kind": "stopped"})
    except Exception:
        results.put({"kind": "error", "message": traceback.format_exc()})
    finally:
        session.close()


class LiveController:
    """No GUI imports, usable in tests and in a Qt timer callback."""
    def __init__(self, scenario, policy="rule", *, native=False, save_path=None):
        context = mp.get_context("spawn")
        self.commands, self.results = context.Queue(), context.Queue()
        self.process = context.Process(target=_worker,
                                       args=(self.commands, self.results, scenario, policy, native, save_path),
                                       name="had-workbench-debug")
        self.process.start()
        self._closed = False

    def send(self, action, **values):
        if not self._closed:
            self.commands.put(dict(action=action, **values))

    def poll(self):
        rows = []
        try:
            while True:
                rows.append(self.results.get_nowait())
        except Empty:
            return rows

    def close(self):
        if self._closed:
            return
        self.send("stop")
        self.process.join(2)
        if self.process.is_alive():
            # This exact handle belongs solely to our debug worker.
            self.process.terminate()
            self.process.join(2)
        self._closed = True
        for queue in (self.commands, self.results):
            queue.cancel_join_thread()
            queue.close()


def _branch_worker(results, document, step, policy, seed):
    try:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
        import sys
        if "torch" in sys.modules:
            sys.modules["torch"].set_num_threads(1)
        from .cli import reconstruct_branch
        from .recording import ReplayEpisode
        episode = ReplayEpisode(document["metadata"], document["frames"], document["decisions"], document.get("sparse", False))
        results.put({"kind": "branch", "episode": reconstruct_branch(episode, step, policy=policy, seed=seed).to_dict()})
    except Exception:
        results.put({"kind": "error", "message": traceback.format_exc()})


class BranchController(LiveController):
    def __init__(self, episode, step, *, policy="grand", seed=20260909):
        context = mp.get_context("spawn")
        self.commands, self.results = context.Queue(), context.Queue()
        self.process = context.Process(target=_branch_worker,
                                       args=(self.results, episode.to_dict(), step, policy, seed),
                                       name="had-workbench-branch")
        self._closed = False
        self.process.start()
