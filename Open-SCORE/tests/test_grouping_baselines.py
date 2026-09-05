from dataclasses import replace
from pathlib import Path
import copy

import numpy as np
import pytest
import torch

from open_score.grouping.baselines import (
    ActionValueNetwork, AlmaStylePolicy, DLOMSearchPolicy, FrozenDLOM,
    StaticPolicy, balanced_initial, candidates,
)
from open_score.grouping.domain import DecisionState, Entity, Group, Grouping


def make_state(red=4):
    return DecisionState(
        0, 50, "reactive",
        tuple(Entity(i, (-1700.0, -500.0 + 300 * i, 100.0), (0.0, 0.0, 0.0), 1.0) for i in range(red)),
        tuple(Entity(20 + i, (1000.0, -400.0 + 250 * i, 100.0), (-100.0, 0.0, 0.0), 1.0) for i in range(4)),
        (Entity(0, (-2100.0, -650.0, 100.0), (0.0, 0.0, 0.0), 1.2),
         Entity(1, (-2100.0, 650.0, 100.0), (0.0, 0.0, 0.0), 1.2)),
        Grouping((), tuple(range(red))),
    )


def test_static_initializes_reserved_roster_then_only_prunes_casualties():
    state = make_state(8)
    policy = StaticPolicy()
    first = policy.act(state).action
    assert not first.reserve
    assert sorted(len(g.members) for g in first.groups) == [4, 4]
    survivor = replace(state, step=5, previous=first,
                       red=tuple(replace(x, health=0.0) if x.id in (0, 6) else x for x in state.red))
    following = policy.act(survivor)
    assert following.action == first.prune(survivor.ids("red"))
    assert following.released_ids == ()
    following.action.validate(survivor.ids("red"), survivor.ids("targets"))


def test_candidate_pool_contains_previous_and_real_same_task_partition_changes():
    state = make_state(8)
    original = balanced_initial(state)
    state = replace(state, step=5, previous=original)
    pool = candidates(state, limit=16, rng=np.random.default_rng(17))
    assert pool[0] == original
    assert len(pool) == len(set(pool)) == 16
    same_assignment = [x for x in pool if x.assignment() == original.assignment()]
    assert len(same_assignment) >= 3
    for action in pool:
        action.validate(state.ids("red"), state.ids("targets"))


def test_dlom_uses_known_distribution_expectation_instead_of_worst_case(monkeypatch):
    import open_score.grouping.baselines as module
    import open_score.grouping.opponents as opponents

    state = make_state()
    red_a = Grouping((Group(0, (0, 1, 2, 3)),))
    red_b = Grouping((Group(1, (0, 1, 2, 3)),))
    blue_a = Grouping((Group(0, (20, 21, 22, 23)),))
    blue_b = Grouping((Group(1, (20, 21, 22, 23)),))
    monkeypatch.setattr(opponents, "distribution", lambda state, name: ([blue_a, blue_b], [0.95, 0.05]))
    # Isolate expected-utility aggregation from the independent matching proxy.
    monkeypatch.setattr(module, "_proxy_pairs", lambda state, red, blue:
                        [(red.groups[0].target, (0,), (20 + blue.groups[0].target,))])

    class Predictor:
        def predict(self, state, requests, style):
            return [({(0, 20): 0.99, (0, 21): 0.01, (1, 20): 0.5, (1, 21): 0.5})[(t, b[0])]
                    for t, r, b in requests]

    policy = DLOMSearchPolicy(predictor=Predictor())
    before = state.to_dict()
    values = policy.rank(state, [red_a, red_b])
    assert values[0] > values[1]
    np.testing.assert_allclose(values[0], (0.95 * np.log(0.99) + 0.05 * np.log(0.01)) / 2)
    assert state.to_dict() == before


def test_frozen_dlom_features_match_the_existing_physical_encoder():
    from open_score.grouping.environment import KnownOpponentEnv
    from open_score.stage2.canonical import HADCanonicalizer

    torch.set_num_threads(1)
    environment = KnownOpponentEnv(red=4, blue=4, max_steps=50, seed=37)
    state = environment.reset()
    path = Path(__file__).resolve().parents[1] / "assets/frozen/dlom.pt"
    rng = torch.get_rng_state().clone()
    model = FrozenDLOM(path)
    assert torch.equal(torch.get_rng_state(), rng)
    encoded = model.encode(state, state.targets[0], state.ids("red"), state.ids("blue"))
    reference = HADCanonicalizer().to_entity_set(environment.adapter.local_state_entities(
        state.targets[0].id, state.ids("red"), state.ids("blue"), local_step=state.step))
    for field in ("target", "red_entities", "blue_entities", "context"):
        np.testing.assert_allclose(getattr(encoded, field), getattr(reference, field), atol=2e-7)
    values = model.predict(state, [(0, state.ids("red"), state.ids("blue"))], "rush")
    assert len(values) == 1 and 0 <= values[0] <= 1
    with pytest.raises(ValueError, match="50-step"):
        model.predict(replace(state, max_steps=40), [(0, (0,), (20,))], "rush")


def test_dlom_search_runs_on_real_state_without_changing_physical_snapshot():
    from open_score.grouping.environment import KnownOpponentEnv

    torch.set_num_threads(1)
    environment = KnownOpponentEnv(red=4, blue=4, seed=27)
    state = environment.reset()
    physical_before = environment.state().to_dict()
    decision = DLOMSearchPolicy(limit=8).act(state)
    decision.action.validate(state.ids("red"), state.ids("targets"))
    assert torch.isfinite(decision.value)
    assert environment.state().to_dict() == physical_before


def test_action_q_is_entity_order_invariant_and_uses_group_membership():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    state = make_state()
    action = balanced_initial(state)
    alternative = Grouping(tuple(Group(g.target, (i,)) for g in action.groups for i in g.members))
    network = ActionValueNetwork(hidden_dim=16, heads=4, layers=1).eval()
    reverse = replace(state, red=tuple(reversed(state.red)), blue=tuple(reversed(state.blue)),
                      targets=tuple(reversed(state.targets)))
    with torch.no_grad():
        value, reordered, split = network([state, reverse, state], [action, action, alternative])
    torch.testing.assert_close(value, reordered, atol=1e-6, rtol=1e-5)
    assert abs(float(value - split)) > 1e-7


def test_alma_replay_updates_q_and_winner_proposal_then_round_trips():
    torch.manual_seed(9)
    torch.set_num_threads(1)
    state = make_state()
    kwargs = dict(hidden_dim=16, heads=4, layers=1, num_candidates=2,
                  batch_size=2, update_every=2, seed=4)
    policy = AlmaStylePolicy(**kwargs)
    q_before = [p.detach().clone() for p in policy.q.parameters()]
    proposal_before = [p.detach().clone() for p in policy.proposal.parameters()]
    for index in range(2):
        decision = policy.act(state)
        decision.action.validate(state.ids("red"), state.ids("targets"))
        policy.observe(state, decision.action, float(index == 1), 5, True, None)
        if index == 0:
            assert policy.update() == {}
    metrics = policy.update()
    assert metrics["updates"] == 1 and metrics["winner_traces"] == 2
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(q_before, policy.q.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(proposal_before, policy.proposal.parameters()))
    assert all(p.grad is None for p in policy.target.parameters())
    assert policy.update() == {}
    saved = copy.deepcopy(policy.training_state_dict())
    expected = policy.act(state)
    restored = AlmaStylePolicy(**kwargs)
    restored.load_training_state_dict(saved)
    actual = restored.act(state)
    assert actual.action == expected.action
    torch.testing.assert_close(actual.value, expected.value)
    assert len(restored.replay) == 2 and restored.observed == 2


def test_alma_nonterminal_target_bootstraps_without_treating_decode_steps_as_time():
    torch.set_num_threads(1)
    state = make_state()
    policy = AlmaStylePolicy(hidden_dim=16, heads=4, layers=1, num_candidates=2,
                             batch_size=2, update_every=2, seed=15)
    for _ in range(2):
        decision = policy.act(state)
        successor = replace(state, step=5, previous=decision.action)
        policy.observe(state, decision.action, 0.0, 5, False, successor)
    result = policy.update()
    assert result["updates"] == 1 and np.isfinite(result["q_loss"])
    with pytest.raises(ValueError, match="physical time"):
        policy.observe(state, balanced_initial(state), 0.0, 0, False, state)
