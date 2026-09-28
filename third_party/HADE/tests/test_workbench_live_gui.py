"""End-to-end Qt CLI smoke tests against a real, owned spawn worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from had_env.workbench import live
from had_env.workbench.cli import main
from had_env.workbench.gui import WorkbenchWindow
from had_env.workbench.recording import load_episode


PROJECT = Path(__file__).resolve().parents[1]
QA = PROJECT / "outputs" / "workbench_qa"


def test_live_cli_qt_controls_and_owned_worker_shutdown(monkeypatch):
    """Real windows, timers, queues and physics; only collect controller evidence."""
    QA.mkdir(parents=True, exist_ok=True)
    output = QA / f"live-gui-partial-{uuid.uuid4().hex}.json.gz"
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(True)
    actual_controller = live.LiveController
    controllers, messages, errors, checkpoints = [], [], [], []

    def observe_controller(*args, **kwargs):
        controller = actual_controller(*args, **kwargs)
        actual_poll = controller.poll

        def poll():
            rows = actual_poll()
            messages.extend(rows)
            return rows

        controller.poll = poll
        controllers.append(controller)
        return controller

    monkeypatch.setattr(live, "LiveController", observe_controller)
    stage = "initial"
    stage_start = started = time.monotonic()
    boundary = 0
    baseline_updates = 0
    timer = QTimer()
    timer.setInterval(25)

    def advance():
        nonlocal stage, stage_start, boundary, baseline_updates
        windows = [window for window in app.topLevelWidgets()
                   if isinstance(window, WorkbenchWindow) and window.isVisible()]
        window = windows[-1] if windows else None
        try:
            assert time.monotonic() - started < 45, f"Live GUI timed out at {stage}"
            assert not [row for row in messages if row["kind"] == "error"], messages
            if window is None or window.episode is None:
                return
            step = window.episode.max_step
            now = time.monotonic()
            if stage == "initial":
                assert step == 0 and not window.playing
                assert controllers[0].process.is_alive()
                stage, stage_start = "hold_initial", now
            elif stage == "hold_initial" and now-stage_start > .4:
                assert step == 0, "Initial paused worker advanced without a command"
                checkpoints.append({"state": "initial_paused", "step": step})
                window.step_button.click()
                stage = "first_step"
            elif stage == "first_step" and step >= 1:
                assert step == 1
                stage, stage_start = "hold_first", now
            elif stage == "hold_first" and now-stage_start > .35:
                assert step == 1
                checkpoints.append({"state": "first_physical_step", "step": step})
                window.step_button.click()
                stage = "second_step"
            elif stage == "second_step" and step >= 2:
                assert step == 2
                stage, stage_start = "hold_second", now
            elif stage == "hold_second" and now-stage_start > .35:
                assert step == 2
                checkpoints.append({"state": "second_physical_step", "step": step})
                window.speed_combo.setCurrentText("8×")
                window.next_decision()
                stage = "decision_boundary"
            elif stage == "decision_boundary" and any(row["kind"] == "paused" for row in messages):
                assert step > 2 and not window.playing
                boundary = step
                stage, stage_start = "hold_boundary", now
            elif stage == "hold_boundary" and now-stage_start > .4:
                assert step == boundary, "Next-decision command did not stop at its boundary"
                checkpoints.append({"state": "next_decision_paused", "step": step})
                baseline_updates = sum(row["kind"] == "update" for row in messages)
                window.play_button.click()
                stage = "play_across_update"
            elif stage == "play_across_update":
                update_count = sum(row["kind"] == "update" for row in messages)
                if update_count > baseline_updates and step > boundary + 1:
                    assert window.playing, "An episode update reset the GUI and paused its live worker"
                    checkpoints.append({"state": "playing_across_update", "step": step})
                    window.pause()
                    stage, stage_start = "hold_pause", now
            elif stage == "hold_pause" and now-stage_start > .5:
                assert not window.playing and not window.episode.metadata.get("complete")
                boundary = step
                stage, stage_start = "verify_pause", now
            elif stage == "verify_pause" and now-stage_start > .35:
                assert step == boundary
                checkpoints.append({"state": "paused_before_close", "step": step})
                window.grab().save(str(QA / "live-gui-verified.png"))
                stage = "closed"
                timer.stop()
                window.close()
        except BaseException as error:
            errors.append(error)
            timer.stop()
            if window is not None:
                window.close()
            app.quit()

    timer.timeout.connect(advance)
    timer.start()
    try:
        result = main(["live", "--red", "8", "--blue", "8", "--seed", "20260907",
                       "--opponent-seed", "20260908", "--output", str(output)])
    finally:
        timer.stop()
        for controller in controllers:
            controller.close()
        for window in app.topLevelWidgets():
            if isinstance(window, WorkbenchWindow):
                window.close()
    assert not errors, repr(errors)
    assert result == 0 and stage == "closed"
    assert len(controllers) == 1
    assert not controllers[0].process.is_alive()
    assert controllers[0].process.exitcode == 0, "Graceful stop should save and exit without terminate()"
    episode = load_episode(output)
    assert episode.max_step == boundary and not episode.metadata["complete"]
    assert [frame["step"] for frame in episode.frames] == list(range(boundary+1))
    assert episode.metadata["scenario"]["opening_seed"] == 20260907
    assert episode.metadata["dt"] == 1
    assert all(frame["sim_time"] == frame["step"] for frame in episode.frames)
    evidence = dict(checkpoints=checkpoints, output=str(output), worker_pid=controllers[0].process.pid,
                    worker_exitcode=controllers[0].process.exitcode, metadata_dt=episode.metadata["dt"],
                    physical_steps=episode.max_step, complete=episode.metadata["complete"])
    (QA / "live-gui-integration.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def test_fresh_native_process_uses_worktree_and_true_clock():
    """Validate launcher-equivalent cwd/PYTHONPATH in a fresh interpreter."""
    QA.mkdir(parents=True, exist_ok=True)
    code = """
import json,sys
import had_env.core.config as config
import had_env
from had_env.workbench.agent_session import FlightSession,FlightScenarioSpec
session=FlightSession(FlightScenarioSpec(seed=71,max_steps=2))
episode=session.run()
session.close()
print(json.dumps(dict(executable=sys.executable,had_env_path=had_env.__file__,
                     config_path=config.__file__,interval=config.Interval,dt=episode.metadata['dt'],
                     seed=episode.metadata['scenario']['seed'],
                     frames=[[f['step'],f['sim_time']] for f in episode.frames])))
"""
    env = dict(os.environ, PYTHONPATH=str(PROJECT), PYGAME_HIDE_SUPPORT_PROMPT="1")
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, env=env,
                               capture_output=True, text=True, check=True, timeout=30)
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(evidence["config_path"]).resolve().is_relative_to(PROJECT)
    assert Path(evidence["had_env_path"]).resolve().is_relative_to(PROJECT)
    assert evidence["interval"] == evidence["dt"] == 1
    assert evidence["seed"] == 71
    assert evidence["frames"] == [[0, 0], [1, 1], [2, 2]]
    (QA / "fresh-native-path-clock.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
