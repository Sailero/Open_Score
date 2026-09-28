"""Interaction tests use recorded truth, not screenshots or physics mocks."""
import copy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from had_env.workbench.gui import WorkbenchWindow
from had_env.workbench.recording import ReplayEpisode


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def recording(sparse=False, divergent=False):
    frames = []
    steps = [0, 3, 8] if sparse else list(range(9))
    for step in steps:
        entities = [
            dict(id=0, entity_id=0, side="red", role="Attack", position=[-800+step*50, 150, 180], velocity=[50, 0, 0], health=1., max_health=1., alive=True, attack_range=500.),
            dict(id=0, entity_id=1, side="blue", role="Scout", position=[700-step*30, -150, 350], velocity=[-30, 0, 0], health=1. if step < 8 else 0., max_health=1., alive=step < 8),
            dict(id=0, entity_id=2, side="targets", role="Entity", position=[-900, -600, 100], velocity=[0, 0, 0], health=4., max_health=5., alive=True),
        ]
        frames.append(dict(step=step, sim_time=step*.1, entities=entities,
                           groups={"red": [{"target_id": 0, "members": [0]}], "blue": []},
                           assignments={"red": {"0": 0}, "blue": {}},
                           actions={"red": {"0": 0}, "blue": {"0": 0}},
                           events=[dict(kind="death", source_id=0, target_id=1, source_side="red", target_side="blue")] if step == 8 else []))
    decisions = [dict(step=0, selected_plan={"members": [0]}, trace={"score": .7}),
                 dict(step=3, selected_plan={"members": [] if divergent else [0]}, trace={"score": .9})]
    return ReplayEpisode(dict(episode_id="test", dt=.1, protocol_id="fixture", physics_version="test-only",
                              scenario={"red_count": 1, "blue_count": 1, "opening_seed": 7, "opponent_seed": 9},
                              policies={"red": "rule", "blue": "reactive"}, outcome_red="win", termination_reason="targets_destroyed"),
                         frames, decisions, sparse=sparse)


@pytest.fixture
def window(app):
    window = WorkbenchWindow(recording())
    window.show()
    app.processEvents()
    yield window
    window.close()
    app.processEvents()


def test_select_identity_physical_step_events_and_terminal(window, app):
    original = copy.deepcopy(window.episode.to_dict())
    assert len(window.xy_view.entities) == 3  # All sides intentionally share id 0.
    window.select_entity("targets", "0")
    assert window.inspector_title.text().startswith("TARGETS 0")
    window.step_forward()
    assert window._frame["step"] == 1
    window.next_decision()
    assert window._frame["step"] == 3
    assert '0.9' in window.decision_summary.toPlainText()
    window.decision_tabs.setCurrentIndex(1)
    assert '0.9' in window.decision_text.toPlainText()
    window.next_event()
    assert window._frame["step"] == 8
    assert window.xy_view.entities[("blue", "0")].flash
    assert not window.xy_view.entities[("targets", "0")].flash
    window.step_forward()
    assert window._frame["step"] == 8
    window.timeline.setValue(0)
    assert window._frame["step"] == 0
    assert window.episode.to_dict() == original


def test_sparse_replay_never_fabricates_frame_or_death(window, app):
    episode = recording(sparse=True)
    episode.frames[0]["entities"][0].update(health=None, alive=None, velocity=None)
    window.set_episode(episode)
    window.seek(2)
    assert window._frame["step"] == 0
    assert "真实采样步 0" in window.statusBar().currentMessage()
    window.step_forward()
    assert window._frame["step"] == 3
    window.seek(0)
    window.select_entity("red", "0")
    app.processEvents()
    assert window.xy_view.grab().width() > 0


def test_comparison_aligns_physical_time_and_reports_divergence(window):
    other = recording(divergent=True)
    other.metadata["dt"] = .05
    for frame in other.frames:
        frame["sim_time"] = frame["step"] * .05
    window.set_comparison(other)
    window.seek(3)
    assert window.compare_view.frame["step"] == 6
    assert window._comparison_divergence is not None
    assert "首次决策分歧" in window.comparison_label.text()
    window.set_comparison(None)
    assert window.compare_view.isHidden()


def test_library_filters_and_live_controls(window):
    episode = recording()
    window.add_episode(episode, "first")
    other = recording()
    other.metadata["policies"]["red"] = "search"
    other.metadata["outcome_red"] = "loss"
    window.add_episode(other, "second")
    window.filter_text.setText("search")
    assert window.episode_list.item(0).isHidden()
    assert not window.episode_list.item(1).isHidden()
    window.outcome_filter.setCurrentIndex(1)
    assert window.episode_list.item(1).isHidden()
    actions = []
    window.control_requested.connect(actions.append)
    window.set_live_mode(True)
    window.toggle_play()
    window.pause()
    window.step_forward()
    window.speed_combo.setCurrentText("2×")
    window.branch_button.click()
    assert actions == ["play", "pause", "step", "speed:2", "branch"]
    new = copy.deepcopy(episode.frames[-1])
    new["step"] = 9
    window.append_frame(new)
    assert window._frame["step"] == 9
    with pytest.raises(ValueError):
        window.append_frame(episode.frames[0])


def test_native_outcome_truncation_and_snapshot_branch_availability(window):
    episode = recording()
    episode.metadata.update(outcome_red=-1, success_native=False)
    episode.decisions[0]["snapshot"] = {"schema": "test"}
    window.set_episode(episode)
    assert window._outcome(episode) == 2
    assert window.branch_button.isEnabled()
    episode.metadata.update(outcome_red=0, truncated=True, termination_reason="sampling_horizon")
    assert window._outcome(episode) == 4
    episode.sparse = True
    window.set_episode(episode)
    assert not window.branch_button.isEnabled()


def test_exports_are_nonempty_and_preserve_theme_and_state(window, tmp_path):
    original = copy.deepcopy(window.episode.to_dict())
    window.select_entity("red", "0")
    for layer in window.layer_actions.values():
        layer.setChecked(True)
    for suffix in ("png", "svg", "pdf"):
        output = tmp_path / f"scene.{suffix}"
        assert window.export_scene(output, light=True) == output
        assert output.stat().st_size > 1000
    assert not window.xy_view.light
    assert window.episode.to_dict() == original
    assert window._frame["step"] == 0


def test_click_select_and_close_observer_do_not_stop_owner(window, app):
    item = window.xy_view.entities[("blue", "0")]
    location = window.xy_view.mapFromScene(item.scenePos())
    QTest.mouseClick(window.xy_view.viewport(), Qt.MouseButton.LeftButton, pos=location)
    app.processEvents()
    assert window._selected == ("blue", "0")
    actions = []
    window.control_requested.connect(actions.append)
    window.set_live_mode(True)
    window.close()
    assert actions == []
    assert not window.timer.isActive()
