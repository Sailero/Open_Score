import copy
from dataclasses import make_dataclass
import json
import pickle

import numpy as np
import pytest

from had_env.grouping.actions import rule_grouping
from had_env.workbench.recording import load_episode, serializable
from had_env.workbench.session import SimulationSession, branch_from_snapshot
from had_env.workbench.snapshot import decode_snapshot, encode_snapshot


def test_json_snapshot_preserves_types_rng_and_exact_future_transitions():
    with SimulationSession() as session:
        session.step()
        before = pickle.dumps(session.snapshot())
        ambient = pickle.dumps(np.random.get_state())
        document = json.loads(json.dumps(encode_snapshot(session.snapshot()), allow_nan=False))
        restored = decode_snapshot(document)
        assert type(restored["physical"].entity_states[0]["Health"]) is type(session.snapshot()["physical"].entity_states[0]["Health"])
        assert pickle.dumps(np.random.get_state()) == ambient
        env = branch_from_snapshot(restored)
        try:
            assert pickle.dumps(session.snapshot()) == before
            while not session.done:
                action = rule_grouping(session.state)
                reference = env.step(action)
                actual = session.step(action)
                assert reference[0].to_dict() == actual.state.to_dict()
                assert reference[3] == actual.info
        finally:
            env.close()


def test_initial_snapshot_and_branch_of_recorded_branch_resume_exactly(tmp_path):
    with SimulationSession() as session:
        initial = decode_snapshot(session.episode.metadata["initial_snapshot"])
        assert initial["physical"].step_count == 0
        session.step()
        with session.branch(728) as branch:
            branch.run(max_decisions=2)
            path = branch.save(tmp_path / "branch.json.gz")
            loaded = load_episode(path)
            decision = loaded.decisions[1]
            with SimulationSession.from_snapshot(decision["snapshot"], loaded.metadata["scenario"]) as reopened:
                assert serializable(reopened.state.to_dict()) == decision["state"]
                actual = reopened.step(decision["selected_plan"])
                assert actual.state.step == decision["step"] + decision["delta"]
                assert reopened.episode.frames[-1]["entities"] == loaded.frame_at(actual.state.step)["entities"]
                with reopened.branch(934) as nested:
                    assert nested.state.to_dict() == reopened.state.to_dict()
                    assert nested.episode.metadata["initial_snapshot"]["schema"] == "had-workbench-snapshot-v1"


def test_snapshot_rejects_source_changes_executable_types_and_tampering():
    with SimulationSession() as session:
        document = encode_snapshot(session.snapshot())
    wrong = copy.deepcopy(document["source_identity"])
    wrong["behavior_hash"] = "different"
    with pytest.raises(ValueError, match="sources differ"):
        decode_snapshot(document, current_identity=wrong)
    broken = copy.deepcopy(document)
    broken["payload"]["kind"] = "pickle"
    with pytest.raises(ValueError, match="hash mismatch"):
        decode_snapshot(broken)
    from had_env.workbench.snapshot import _hash
    broken["payload_hash"] = _hash(broken["payload"])
    with pytest.raises(ValueError, match="Unregistered"):
        decode_snapshot(broken)


def test_snapshot_rejects_malformed_shapes_before_restore():
    with SimulationSession() as session:
        snapshot = session.snapshot()
        snapshot["physical"].entity_states[0]["position"] = [1., 2.]
        encoded = encode_snapshot(snapshot)
    with pytest.raises(ValueError, match="xyz"):
        decode_snapshot(encoded)


def test_snapshot_cannot_encode_an_unregistered_class_with_a_trusted_name():
    # Name equality alone must never authorize a plugin's class for restoration.
    impostor = make_dataclass("HADStage3Snapshot", [("value", int)])
    with pytest.raises(ValueError, match="Unregistered snapshot dataclass"):
        encode_snapshot({"physical": impostor(123)})
