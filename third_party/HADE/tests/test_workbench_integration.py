import copy
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pytest

from had_env.workbench.agent_session import FlightScenarioSpec, FlightSession
from had_env.workbench.cli import main
from had_env.workbench.identity import source_identity
from had_env.workbench.playback import display_frame
from had_env.workbench.recording import ReplayEpisode, load_episode


def test_installed_cli_defaults_write_to_calling_directory_not_site_packages(tmp_path, monkeypatch):
    from had_env.workbench.cli import _output_root
    caller = tmp_path / "experiment"
    caller.mkdir()
    monkeypatch.chdir(caller)
    installed = tmp_path / "site-packages" / "had_env" / "workbench" / "cli.py"
    assert _output_root(installed) == caller
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'had-env'\n")
    assert _output_root(checkout / "had_env" / "workbench" / "cli.py") == checkout


def test_native_heterogeneous_two_sided_actions_and_branch():
    session = FlightSession(FlightScenarioSpec(red_attackers=2, blue_attackers=2, red_scouts=1,
                                               blue_disturbers=1, max_steps=4))
    try:
        before = pickle.dumps(session.snapshot())
        with pytest.raises(ValueError):
            session.step({"red": [0, 0, 0], "blue": [99, 0, 0]})
        assert before == pickle.dumps(session.snapshot())
        session.step({"red": [0, 0, 0], "blue": [0, 0, 0]})
        child = session.branch()
        try:
            first, second = session.run(), child.run()
            assert first.frames == second.frames
            assert first.metadata["truncated"] and not first.metadata["terminated"]
            assert first.metadata["success_native"] is None
            assert len(first.frames) == 5
            assert {e["role"] for e in first.frames[0]["entities"]} >= {"Attack", "Scout", "Disturb"}
            assert first.frames[1]["actions"]["red"] and first.frames[1]["actions"]["blue"]
            with pytest.raises(RuntimeError):
                session.branch()
        finally:
            child.close()
    finally:
        session.close()
    with pytest.raises(RuntimeError):
        session.step()


def test_two_policies_receive_precommit_observations_without_engine_access():
    seen = []
    class Policy:
        def act(self, obs):
            assert set(obs) == {"observation", "agent_ids", "alive_mask"}
            seen.append(copy.deepcopy(obs))
            obs["observation"].clear()
            return [0] * len(obs["agent_ids"])
    session = FlightSession(FlightScenarioSpec(red_attackers=1, blue_attackers=1), red_policy=Policy(), blue_policy=Policy())
    try:
        before = session.observations()
        session.step()
        assert seen == [before["red"], before["blue"]]
    finally:
        session.close()


def test_native_stochastic_policy_owns_memory_and_all_ambient_random_streams():
    import random
    torch = pytest.importorskip("torch")
    from had_env.workbench.rng import random_state
    class Stochastic:
        def __init__(self):
            self.calls = 0

        def act(self, observation):
            self.calls += 1
            return [int(torch.randint(0, 27, ()).item() + random.randrange(27) + np.random.randint(27)) % 27
                    for _ in observation["agent_ids"]]
    policy = Stochastic()
    before = random_state()
    session = FlightSession(FlightScenarioSpec(red_attackers=1, blue_attackers=1, max_steps=3), red_policy=policy)
    child = session.branch()
    try:
        assert session.run().frames == child.run().frames
        assert policy.calls == 0
        after = random_state()
        assert before["python"] == after["python"]
        assert np.array_equal(before["numpy"][1], after["numpy"][1])
        assert torch.equal(before["torch"], after["torch"])
    finally:
        session.close()
        child.close()


def test_display_interpolation_never_changes_health_events_or_original_recording():
    a = dict(step=0, entities=[dict(id=0, side="red", alive=True, position=[0, 0, 0], health=1.)], events=[])
    b = dict(step=1, entities=[dict(id=0, side="red", alive=True, position=[10, 0, 0], health=.2)], events=[{"kind": "damage"}])
    episode = ReplayEpisode({}, [a, b], [])
    result = display_frame(episode, .5)
    assert result["entities"][0]["position"] == [5, 0, 0]
    assert result["entities"][0]["health"] == 1. and result["events"] == []
    assert episode.frames[0] == a
    episode.sparse = True
    assert display_frame(episode, .5) == a


def test_viewer_edits_change_provenance_without_claiming_new_physics(tmp_path):
    source = tmp_path / "had_env" / "workbench"
    source.mkdir(parents=True)
    (source / "gui.py").write_text("VIEW = 1")
    (source / "session.py").write_text("STATE = 1")
    first = source_identity(tmp_path)
    (source / "gui.py").write_text("VIEW = 2")
    second = source_identity(tmp_path)
    assert first["behavior_hash"] == second["behavior_hash"]
    assert first["viewer_hash"] != second["viewer_hash"]
    assert first["source_hash"] != second["source_hash"]
    (source / "session.py").write_text("STATE = 2")
    assert second["behavior_hash"] != source_identity(tmp_path)["behavior_hash"]
    before_entry = source_identity(tmp_path)
    (tmp_path / "make_env.py").write_text("FACTORY_PROTOCOL = 1")
    assert before_entry["behavior_hash"] != source_identity(tmp_path)["behavior_hash"]


def test_cli_records_loadable_complete_episode_and_exports_figures(tmp_path):
    recording = tmp_path / "episode.json.gz"
    assert main(["record", "--red", "2", "--blue", "2", "--output", str(recording)]) == 0
    episode = load_episode(recording)
    assert episode.metadata["complete"] and len(episode.frames) == episode.max_step + 1
    for suffix in ("png", "svg", "pdf"):
        target = tmp_path / f"figure.{suffix}"
        assert main(["export", str(recording), str(target)]) == 0
        assert target.stat().st_size > 1000
        assert json.loads(Path(str(target)+".json").read_text())["recording_metadata"]["scenario"]


def test_video_decodes_with_requested_dimensions_and_final_frame(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio
    from had_env.workbench.export import export_video
    session = FlightSession(FlightScenarioSpec(red_attackers=1, blue_attackers=1, max_steps=2))
    try:
        episode = session.run()
    finally:
        session.close()
    # A saved continuation can start at a nonzero physical step; no invented lead-in.
    for frame in episode.frames:
        frame["step"] += 5
        frame["sim_time"] += 5
    target = export_video(episode, tmp_path / "episode.mp4", fps=2, steps_per_second=2, width=640, height=480)
    with imageio.get_reader(target) as reader:
        frames = list(reader.iter_data())
    assert len(frames) == 3 and frames[-1].shape == (480, 640, 3)
    assert not np.array_equal(frames[0], frames[-1])


def test_cli_saved_decision_can_branch_again_without_reconstructing_unknown_history(tmp_path):
    from had_env.workbench.protocols import ScenarioSpec
    from had_env.workbench.session import SimulationSession
    original = tmp_path / "original.json.gz"
    with SimulationSession(ScenarioSpec()) as session:
        episode = session.run()
        session.save(original)
    step = episode.decisions[1]["step"]
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    main(["branch", str(original), "--step", str(step), "--policy", "grand", "--output", str(first)])
    main(["branch", str(first), "--step", str(step), "--policy", "rule", "--output", str(second)])
    assert load_episode(first).frames[0]["entities"] == episode.frame_at(step)["entities"]
    assert load_episode(second).frames[0]["entities"] == episode.frame_at(step)["entities"]
    assert load_episode(second).metadata["complete"]


def test_cli_explicit_split_evaluation_writes_records_and_refuses_overwrite(tmp_path):
    from had_env.workbench.protocols import scenario_family
    from had_env.workbench.evaluation import EvaluationPlan
    plan = EvaluationPlan(
        train=(scenario_family(10, 0, split="train", cells=((2, 2),)),),
        validation=(), test=(scenario_family(11, 0, split="test", cells=((2, 2),)),))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.to_dict()))
    destination = tmp_path / "evaluation"
    assert main(["evaluate", str(path), "--output", str(destination)]) == 0
    rows = json.loads((destination / "episodes.json").read_text())
    assert len(rows) == 2 and {r["method_id"] for r in rows} == {"rule", "grand"}
    assert all(load_episode(r["recording"]).metadata["complete"] for r in rows)
    with pytest.raises((ValueError, FileExistsError)):
        main(["evaluate", str(path), "--output", str(destination)])


def _wait_message(controller, wanted, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rows = controller.poll()
        errors = [r for r in rows if r["kind"] == "error"]
        assert not errors, errors
        selected = [r for r in rows if r["kind"] == wanted]
        if selected:
            return selected[-1]
        assert controller.process.is_alive(), f"Worker exited {controller.process.exitcode}"
        time.sleep(.02)
    raise AssertionError(f"No {wanted} received from owned debug worker")


def test_spawn_debug_worker_pauses_and_advances_one_physical_step_only():
    from had_env.workbench.live import LiveController
    from had_env.workbench.protocols import ScenarioSpec
    controller = LiveController(ScenarioSpec(red_count=2, blue_count=2).to_dict())
    try:
        initial = _wait_message(controller, "episode")
        assert initial["episode"]["frames"][0]["step"] == 0
        controller.send("step")
        first = _wait_message(controller, "frame")
        assert first["frame"]["step"] == 1
        time.sleep(.12)
        assert not [r for r in controller.poll() if r["kind"] == "frame"]
        controller.send("step")
        assert _wait_message(controller, "frame")["frame"]["step"] == 2
    finally:
        controller.close()
    assert not controller.process.is_alive()
