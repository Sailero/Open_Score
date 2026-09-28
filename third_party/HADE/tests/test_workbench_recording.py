import gzip
import json

import pytest

from had_env.workbench.recording import EpisodeRecorder, load_episode
from had_env.workbench.session import SimulationSession


def test_full_recording_roundtrip_captures_every_physical_step_and_terminal(tmp_path):
    with SimulationSession() as session:
        episode = session.run()
        assert episode.metadata["complete"] and not episode.sparse
        assert [f["step"] for f in episode.frames] == list(range(episode.max_step + 1))
        assert len(episode.frames[0]["entities"]) == 18
        assert all(len(f["entities"]) == 18 for f in episode.frames)
        assert episode.frames[0]["actions"] == {"red": {}, "blue": {}}
        assert all(len(f["actions"]["red"]) == 8 and len(f["actions"]["blue"]) == 8
                   for f in episode.frames[1:])
        assert episode.frames[-1]["events"]
        physics_events = [(frame["step"], event) for frame in episode.frames
                          for event in frame["events"] if event["phase"] == "physics"]
        assert physics_events
        assert all(step == event["step"] for step, event in physics_events)
        assert episode.metadata["outcome_red"] in (-1, 1)
        assert sum(d["delta"] for d in episode.decisions) == episode.max_step
        for suffix in (".json", ".json.gz"):
            path = session.save(tmp_path / ("episode" + suffix))
            loaded = load_episode(path)
            assert loaded.to_dict() == episode.to_dict()
            selected = loaded.frame_at(2)
            selected["entities"][0]["health"] = -999
            assert loaded.frame_at(2)["entities"][0]["health"] != -999


def test_v5_import_keeps_only_recorded_spatial_states(tmp_path):
    path = tmp_path / "family.jsonl.gz"
    rows = [dict(record_type="episode", family_id="f", physical_steps=10, success_native=True),
            dict(record_type="event", family_id="f", physical_step=0,
                 state=dict(step=0, red=[dict(id=1, position=[0, 0, 100], velocity=[1, 0, 0], health=1)],
                            blue=[], targets=[]),
                 selected_plan_raw=dict(groups=[], reserve=[1]), decision_trace={"candidates": [{"y": .5}]},
                 physical_trace=[dict(physical_step=1, target_hp={"0": 1.2}, actions={"1": 0})]),
            dict(record_type="event", family_id="f", physical_step=5,
                 state=dict(step=5, red=[dict(id=1, position=[5, 0, 100], velocity=[1, 0, 0], health=1)],
                            blue=[], targets=[]), selected_plan_raw=dict(groups=[], reserve=[1]))]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("\n".join(map(json.dumps, rows)))
    original = path.read_bytes()
    episode = load_episode(path)
    assert episode.sparse and episode.max_step == 10
    assert [f["step"] for f in episode.frames] == [0, 5]
    assert episode.frame_at(10)["step"] == 5
    assert episode.decisions[0]["trace"]["candidates"] == [{"y": .5}]
    assert episode.decisions[0]["physical_trace"][0]["physical_step"] == 1
    assert path.read_bytes() == original


def test_v4_import_does_not_invent_health_or_velocity(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([dict(step=0, positions={"0": [1, 2, 3]},
                                    targets={"0": [-2100, 0, 100]})]), encoding="utf-8")
    episode = load_episode(path)
    assert episode.sparse
    assert all(e["health"] is None and e["velocity"] is None for e in episode.frames[0]["entities"])
    assert {(e["side"], e["id"]) for e in episode.frames[0]["entities"]} == {("red", 0), ("targets", 0)}


def test_invalid_new_frame_order_is_rejected(tmp_path):
    recorder = EpisodeRecorder()
    recorder.append_frame(dict(step=2, entities=[]))
    with pytest.raises(ValueError, match="strictly"):
        recorder.append_frame(dict(step=1, entities=[]))
    payload = recorder.episode.to_dict()
    payload["frames"].append(dict(step=2, entities=[]))
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_episode(path)
