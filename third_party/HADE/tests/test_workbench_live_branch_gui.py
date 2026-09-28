"""Live branching goes through the real CLI, GUI and owned spawn worker."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time
import uuid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from had_env.workbench import cli, live
from had_env.workbench.gui import WorkbenchWindow
from had_env.workbench.recording import load_episode


def test_live_branch_is_saved_without_advancing_parent(monkeypatch):
    qa = Path(__file__).resolve().parents[1] / "outputs" / "workbench_qa"
    run_root = qa / f"live-branch-{uuid.uuid4().hex}"
    run_root.mkdir(parents=True)
    parent_output = run_root / "parent-partial.json.gz"
    # Keep all test recordings in the QA directory; the real CLI still selects
    # and writes its own timestamped branch filename beneath WORKBENCH_ROOT.
    monkeypatch.setattr(cli, "WORKBENCH_ROOT", run_root)
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(True)
    actual_controller = live.LiveController
    controllers, messages, errors = [], [], []

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
    stage, started = "initial", time.monotonic()
    held_at = None
    parent_frame, boundary, child = None, None, None
    timer = QTimer()
    timer.setInterval(25)

    def advance():
        nonlocal stage, held_at, parent_frame, boundary, child
        windows = [window for window in app.topLevelWidgets()
                   if isinstance(window, WorkbenchWindow) and window.isVisible()]
        window = windows[-1] if windows else None
        try:
            assert time.monotonic()-started < 40, f"Live branch timed out at {stage}"
            assert not [row for row in messages if row["kind"] == "error"], messages
            if window is None or window.episode is None:
                return
            if stage == "initial":
                assert window.episode.max_step == 0 and not window.playing
                assert window.branch_button.isEnabled()
                window.branch_button.click()
                window.speed_combo.setCurrentText("8×")
                window.next_decision()
                stage = "await_branch"
            elif stage == "await_branch" and window.comparison is not None:
                assert any(row["kind"] == "branch" for row in messages)
                assert any(row["kind"] == "paused" for row in messages)
                updates = [row["episode"] for row in messages if row["kind"] == "update"]
                parent_frame = copy.deepcopy(updates[-1]["frames"][-1])
                boundary = parent_frame["step"]
                child = window.comparison
                assert boundary == 5
                assert window.episode.max_step == boundary
                assert window.episode.frames[-1] == parent_frame
                assert child.metadata["branch"]["parent_step"] == boundary
                assert child.metadata["branch"]["continuation_seed"] == 20260909
                assert child.frames[0]["entities"] == parent_frame["entities"]
                assert child.max_step > boundary and child.metadata["complete"]
                files = list((run_root / "outputs" / "workbench").glob("live_branch_*.json.gz"))
                assert len(files) == 1
                assert load_episode(files[0]).frames == child.frames
                assert any(episode is child for _, episode in window._episodes)
                held_at, stage = time.monotonic(), "hold_parent"
            elif stage == "hold_parent" and time.monotonic()-held_at > .45:
                assert window.episode.max_step == boundary
                assert window.episode.frames[-1] == parent_frame
                assert max(row["frame"]["step"] for row in messages if row["kind"] == "frame") == boundary
                assert not window.playing and controllers[0].process.is_alive()
                window.step_button.click()
                stage = "resume_parent_one_step"
            elif stage == "resume_parent_one_step" and window.episode.max_step > boundary:
                assert window.episode.max_step == boundary+1
                assert window.episode.frame_at(boundary) == parent_frame
                assert window.comparison is child
                window.grab().save(str(qa / "live-branch-verified.png"))
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
        result = cli.main(["live", "--red", "2", "--blue", "2", "--seed", "20260907",
                           "--opponent-seed", "20260908", "--output", str(parent_output)])
    finally:
        timer.stop()
        for controller in controllers:
            controller.close()
        for window in app.topLevelWidgets():
            if isinstance(window, WorkbenchWindow):
                window.close()
    assert not errors, repr(errors)
    assert result == 0 and stage == "closed"
    assert len(controllers) == 1 and controllers[0].process.exitcode == 0
    parent = load_episode(parent_output)
    assert parent.max_step == boundary+1 and not parent.metadata["complete"]
    assert parent.frame_at(boundary) == parent_frame
    files = list((run_root / "outputs" / "workbench").glob("live_branch_*.json.gz"))
    evidence = dict(parent_output=str(parent_output), branch_output=str(files[0]),
                    parent_branch_step=boundary, parent_saved_step=parent.max_step,
                    branch_start_step=child.frames[0]["step"], branch_final_step=child.max_step,
                    branch_complete=child.metadata["complete"], parent_unchanged_at_branch=True,
                    worker_pid=controllers[0].process.pid, worker_exitcode=controllers[0].process.exitcode)
    (qa / "live-branch-integration.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
