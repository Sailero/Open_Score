import pickle

import numpy as np
import pytest

from had_env.grouping.actions import rule_grouping
from had_env.grouping.rules import make_env
from had_env.workbench.protocols import ScenarioSpec
from had_env.workbench.session import PolicyAdapter, SimulationSession, branch_from_snapshot


def test_recorded_session_matches_native_grouping_every_transition():
    spec = ScenarioSpec(opening_seed=73, opponent_seed=123)
    reference = make_env(spec.red_count, spec.blue_count, opponent=spec.opponent,
                         seed=spec.opening_seed, max_steps=spec.max_steps,
                         command_interval=spec.command_interval,
                         targets=len(spec.target_positions), target_positions=spec.target_positions)
    reference.reset(seed=spec.opening_seed)
    reference.set_rng(spec.opponent_seed)
    try:
        with SimulationSession(spec) as session:
            while not reference.done:
                expected = reference.step(rule_grouping(reference.state()))
                actual = session.step()
                assert actual.state.to_dict() == expected[0].to_dict()
                assert actual.reward == expected[1] and actual.done == expected[2]
                assert actual.delta == expected[3]["delta"]
                assert actual.info["events"] == expected[3]["events"]
                assert actual.terminated == expected[3]["terminated"]
                assert actual.truncated == expected[3]["truncated"]
            assert session.episode.frames[-1]["step"] == reference.state().step
    finally:
        reference.close()


def test_snapshot_branch_is_exact_and_new_rng_does_not_mutate_live_or_ambient():
    with SimulationSession() as session:
        session.step()
        snapshot = session.snapshot()
        before = pickle.dumps(snapshot)
        np.random.seed(188)
        numpy_before = pickle.dumps(np.random.get_state())
        exact = branch_from_snapshot(snapshot)
        independent = branch_from_snapshot(snapshot, continuation_seed=123)
        try:
            assert independent.state().to_dict() == session.state.to_dict()
            assert pickle.dumps(session.snapshot()) == before
            assert pickle.dumps(np.random.get_state()) == numpy_before
            action = rule_grouping(session.state)
            a = exact.step(action)
            b = session.step(action)
            assert a[0].to_dict() == b.state.to_dict()
            assert a[3] == b.info
        finally:
            exact.close()
            independent.close()


class MutatingPlanner:
    def act_env(self, env):
        plan = rule_grouping(env.state())
        env.step(plan)
        env.adapter.env.world[0].Health = -7
        np.random.random(20)
        return plan


def test_external_planner_only_receives_disposable_branch_and_private_rng():
    with SimulationSession() as session:
        before = pickle.dumps(session.snapshot())
        adapter = PolicyAdapter(MutatingPlanner(), seed=91)
        np_before = pickle.dumps(np.random.get_state())
        action = adapter.act(session.env, planning_seed=190)
        action.validate(session.state.ids("red"), session.state.ids("targets"), max_members=None)
        assert pickle.dumps(session.snapshot()) == before
        assert pickle.dumps(np.random.get_state()) == np_before


def test_invalid_action_is_rejected_before_mutating_live_physics():
    with SimulationSession() as session:
        before = pickle.dumps(session.snapshot())
        with pytest.raises(ValueError):
            session.step({"groups": [], "reserve": []})
        assert pickle.dumps(session.snapshot()) == before
        assert len(session.episode.frames) == 1


def test_intervention_branch_marks_parent_and_never_relabels_policy_run():
    with SimulationSession() as session:
        session.step()
        before = pickle.dumps(session.snapshot())
        with session.branch(continuation_seed=728) as branch:
            assert branch.state.to_dict() == session.state.to_dict()
            branch.step(rule_grouping(branch.state))
            assert branch.episode.metadata["contains_interventions"]
            assert branch.episode.metadata["branch"]["parent_step"] == session.state.step
            assert branch.episode.frames[0]["events"] == []
        assert pickle.dumps(session.snapshot()) == before


def test_terminal_session_cannot_create_fictitious_continuation():
    from had_env.workbench.protocols import CUSTOM_PROTOCOL
    with SimulationSession(ScenarioSpec(max_steps=1, protocol_id=CUSTOM_PROTOCOL)) as session:
        session.run()
        with pytest.raises(RuntimeError, match="terminal"):
            session.branch()


def test_interrupted_macro_preserves_command_snapshot_and_real_physical_work(tmp_path, monkeypatch):
    from had_env.workbench.recording import load_episode
    from had_env.workbench.snapshot import decode_snapshot
    class StopExecution(Exception):
        pass
    with SimulationSession() as session:
        original = session.env.adapter.step
        def stop_before_third(*args, **kwargs):
            # The decision is inspectable before even the first physical step.
            assert session.episode.decisions[0]["execution_status"] == "in_progress"
            if session.env.adapter.step_count == 2:
                raise StopExecution()
            return original(*args, **kwargs)
        monkeypatch.setattr(session.env.adapter, "step", stop_before_third)
        with pytest.raises(StopExecution):
            session.step()
        assert session.env.adapter.step is stop_before_third
        assert [frame["step"] for frame in session.episode.frames] == [0, 1, 2]
        assert len(session.episode.decisions) == 1
        decision = session.episode.decisions[0]
        assert decision["execution_status"] == "interrupted" and decision["delta"] == 2
        assert decision["selected_plan"]["groups"]
        assert decode_snapshot(decision["snapshot"])["physical"].step_count == 0
        assert not session.episode.metadata["complete"]
        assert "success_native" not in session.episode.metadata
        path = session.save(tmp_path / "interrupted.json.gz")
        loaded = load_episode(path)
        assert loaded.max_step == 2 and loaded.decisions == session.episode.decisions
        assert loaded.metadata["execution_status"] == "interrupted"
        with pytest.raises(RuntimeError, match="Interrupted macro"):
            session.step()
