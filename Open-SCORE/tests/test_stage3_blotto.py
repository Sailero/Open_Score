import itertools

import numpy as np
import pytest

from open_score.stage3 import (
    BayesianEventBlottoGame,
    EventBlottoGame,
    TypedBluePolicy,
    allocation_marginals,
    make_blue_grouping_type,
    match_agents_to_tasks,
    match_sparse_agents_to_tasks,
    payoff_from_breach_risk,
    solve_double_oracle,
    solve_bayesian_double_oracle,
    solve_restricted_matrix_game,
)


def _two_battlefield_protection_game(allow_reserve: bool = False) -> EventBlottoGame:
    # An unprotected attacked target costs Red one point.  A protected target
    # and every unattacked target contribute zero.
    local = np.zeros((2, 2, 2), dtype=np.float64)
    local[:, 0, 1] = -1.0
    return EventBlottoGame(
        local,
        defender_budget=1,
        attacker_budget=1,
        allow_defender_reserve=allow_reserve,
        allow_attacker_reserve=allow_reserve,
        event_id="unit-test-event",
    )


def test_payoff_from_breach_risk_uses_risk_time_and_weights():
    breach = np.array([[[0.2, 0.8]], [[0.5, 0.25]]], dtype=np.float64)
    pressure = np.full_like(breach, 0.1)
    payoff = payoff_from_breach_risk(
        breach,
        target_weights=[2.0, 1.0],
        early_breach_pressure=pressure,
        breach_loss=1.0,
        survival_reward=0.5,
        time_weight=0.2,
    )
    expected = np.array([[[0.36, -1.44]], [[-0.27, 0.105]]])
    np.testing.assert_allclose(payoff, expected)


def test_restricted_matrix_game_solves_both_players():
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
    solution = solve_restricted_matrix_game(matrix)
    np.testing.assert_allclose(solution.defender_mixture, [0.5, 0.5], atol=1e-7)
    np.testing.assert_allclose(solution.attacker_mixture, [0.5, 0.5], atol=1e-7)
    assert solution.value == pytest.approx(0.0, abs=1e-8)
    assert solution.duality_gap <= 1e-7
    assert not solution.used_fallback


def test_dp_best_responses_use_mixed_strategy_marginals():
    game = _two_battlefield_protection_game()
    attackers = [(1, 0), (0, 1)]
    response = game.defender_best_response(attackers, [0.75, 0.25])
    assert response.allocation == (1, 0)
    assert response.value == pytest.approx(-0.25)
    assert response.reserve_resources == 0

    defenders = [(1, 0), (0, 1)]
    attack_response = game.attacker_best_response(defenders, [0.75, 0.25])
    assert attack_response.allocation == (0, 1)
    assert attack_response.value == pytest.approx(-0.75)

    marginals = allocation_marginals(attackers, [0.75, 0.25], [1, 1])
    np.testing.assert_allclose(marginals, [[0.25, 0.75], [0.75, 0.25]])


@pytest.mark.parametrize("allow_reserve", [False, True])
def test_vectorised_dp_matches_brute_force(allow_reserve):
    rng = np.random.default_rng(1204)
    local = rng.normal(size=(3, 3, 3))
    game = EventBlottoGame(
        local,
        defender_budget=3,
        attacker_budget=2,
        defender_caps=[2, 2, 2],
        attacker_caps=[2, 2, 2],
        allow_defender_reserve=allow_reserve,
        allow_attacker_reserve=allow_reserve,
    )
    attacker_support = [(2, 0, 0), (0, 1, 1)]
    attacker_mixture = [0.35, 0.65]
    response = game.defender_best_response(attacker_support, attacker_mixture)
    feasible = [
        allocation
        for allocation in itertools.product(range(3), repeat=3)
        if (
            sum(allocation) <= game.defender_budget
            if allow_reserve
            else sum(allocation) == game.defender_budget
        )
    ]
    brute_values = [
        sum(
            probability * game.payoff(allocation, attacker)
            for probability, attacker in zip(attacker_mixture, attacker_support)
        )
        for allocation in feasible
    ]
    assert response.value == pytest.approx(max(brute_values))


def test_double_oracle_finds_blotto_mixed_equilibrium_and_samples_reproducibly():
    game = _two_battlefield_protection_game()
    progress = []
    result = solve_double_oracle(
        game,
        max_iterations=10,
        tolerance=1e-9,
        seed=17,
        progress_callback=progress.append,
    )
    assert result.converged
    assert not result.used_matrix_game_fallback
    assert result.termination_reason == "converged"
    assert result.value == pytest.approx(-0.5, abs=1e-7)
    assert result.exploitability <= 1e-8
    assert len(result.defender_strategies) == 2
    assert len(result.attacker_strategies) == 2
    assert len(progress) == len(result.history)
    assert result.sample_profile(seed=91) == result.sample_profile(seed=91)
    assert result.sample_profile() == result.sample_profile()


def test_bayesian_double_oracle_matches_complete_typed_matrix():
    game = _two_battlefield_protection_game()
    balanced = make_blue_grouping_type(
        game,
        name="balanced",
        prior=0.35,
        allocation_family="balanced",
    )
    strategic = make_blue_grouping_type(
        game,
        name="strategic",
        prior=0.65,
        allocation_family="strategic",
    )
    typed = BayesianEventBlottoGame((balanced, strategic))
    defenders = ((1, 0), (0, 1))
    balanced_actions = balanced.candidate_allocations
    assert balanced_actions is not None
    strategic_actions = ((1, 0), (0, 1))
    blue_policies = tuple(
        TypedBluePolicy((fixed, response))
        for fixed in balanced_actions
        for response in strategic_actions
    )
    exact = solve_restricted_matrix_game(
        typed.payoff_matrix(defenders, blue_policies)
    )
    result = solve_bayesian_double_oracle(
        typed, max_iterations=20, tolerance=1e-9, seed=19
    )

    assert result.converged
    assert result.value == pytest.approx(exact.value, abs=1e-8)
    assert result.exploitability <= 1e-8
    sample = result.sample_profile(seed=22, blue_type_name="strategic")
    assert sample.blue_type_name == "strategic"
    assert sample.attacker_allocation in strategic_actions


def test_bayesian_red_distribution_does_not_change_when_conditioning_viewed_type():
    game = _two_battlefield_protection_game()
    typed = BayesianEventBlottoGame(
        (
            make_blue_grouping_type(
                game, name="balanced", prior=0.5, allocation_family="balanced"
            ),
            make_blue_grouping_type(
                game, name="strategic", prior=0.5, allocation_family="strategic"
            ),
        )
    )
    result = solve_bayesian_double_oracle(typed, max_iterations=20, tolerance=1e-9)
    for seed in range(10):
        balanced = result.sample_profile(seed=seed, blue_type_name="balanced")
        strategic = result.sample_profile(seed=seed, blue_type_name="strategic")
        assert balanced.defender_allocation == strategic.defender_allocation


def test_reserve_is_an_implicit_blotto_battlefield():
    local = np.zeros((1, 3, 2), dtype=np.float64)
    # Deploying defenders is costly if Blue leaves this target alone.
    local[0, :, 0] = [0.0, -1.0, -2.0]
    game = EventBlottoGame(
        local,
        defender_budget=2,
        attacker_budget=1,
        allow_defender_reserve=True,
        allow_attacker_reserve=True,
    )
    response = game.defender_best_response([(0,)], [1.0])
    assert response.allocation == (0,)
    assert response.assigned_resources == 0
    assert response.reserve_resources == 2


def test_per_battlefield_caps_can_be_smaller_than_tensor_dimensions():
    local = np.zeros((2, 5, 4), dtype=np.float64)
    local[:, :, 1] = -1.0
    game = EventBlottoGame(
        local,
        defender_budget=2,
        attacker_budget=2,
        defender_caps=[2, 2],
        attacker_caps=[1, 1],
        allow_defender_reserve=False,
        allow_attacker_reserve=False,
    )
    response = game.defender_best_response([(1, 1)], [1.0])
    assert sum(response.allocation) == 2


def test_large_best_response_does_not_enumerate_allocations():
    # The corresponding pure spaces contain astronomically many allocations;
    # the test exercises only O(M * budget * local_cap) DP states.
    n_battlefields = 200
    red_counts = np.arange(5, dtype=np.float64)[None, :, None]
    blue_counts = np.arange(4, dtype=np.float64)[None, None, :]
    local = np.broadcast_to(
        -(blue_counts + 1.0) / (red_counts + 1.0),
        (n_battlefields, 5, 4),
    ).copy()
    game = EventBlottoGame(
        local,
        defender_budget=600,
        attacker_budget=400,
        defender_caps=np.full(n_battlefields, 4),
        attacker_caps=np.full(n_battlefields, 3),
        allow_defender_reserve=True,
        allow_attacker_reserve=True,
        event_id="theory-only-1000",
    )
    attacker = tuple([2] * n_battlefields)
    defender_response = game.defender_best_response([attacker], [1.0])
    assert len(defender_response.allocation) == n_battlefields
    assert sum(defender_response.allocation) <= 600
    assert max(defender_response.allocation) <= 4


def test_double_oracle_does_not_stall_when_two_subtolerance_gaps_sum_above_limit():
    # With a total exploitability tolerance of 0.5, this deterministic game
    # reaches a profile whose two unilateral gaps are each below 0.5 but sum
    # above it.  At least one oracle must still be added (threshold 0.25).
    local = np.random.default_rng(17).normal(size=(3, 3, 3))
    game = EventBlottoGame(
        local,
        defender_budget=3,
        attacker_budget=3,
        defender_caps=np.full(3, 2),
        attacker_caps=np.full(3, 2),
        allow_defender_reserve=True,
        allow_attacker_reserve=True,
    )
    result = solve_double_oracle(game, max_iterations=30, tolerance=0.5, seed=17)
    relevant = [
        row
        for row in result.history
        if row.exploitability > 0.5
        and row.defender_gap <= 0.5
        and row.attacker_gap <= 0.5
    ]
    assert relevant
    assert all(
        row.added_defender_strategy or row.added_attacker_strategy for row in relevant
    )
    assert result.termination_reason == "converged"
    assert result.exploitability <= 0.5


def test_optional_generic_count_mask_is_shared_by_validation_and_oracles():
    local = np.zeros((2, 4, 3), dtype=np.float64)
    local[:, 1, :] = 100.0  # A tempting count forbidden only in this generic API test.
    local[:, 2, :] = 1.0
    defender_mask = np.ones((2, 4), dtype=bool)
    defender_mask[:, 1] = False
    attacker_mask = np.ones((2, 3), dtype=bool)
    attacker_mask[:, 1] = False
    game = EventBlottoGame(
        local,
        defender_budget=3,
        attacker_budget=2,
        defender_caps=[3, 3],
        attacker_caps=[2, 2],
        defender_count_mask=defender_mask,
        attacker_count_mask=attacker_mask,
    )

    with pytest.raises(ValueError, match="unsupported local count"):
        game.validate_defender_allocation((1, 0))
    with pytest.raises(ValueError, match="unsupported local count"):
        game.validate_attacker_allocation((1, 0))
    response = game.defender_best_response([(0, 0)], [1.0])
    assert 1 not in response.allocation
    assert response.assigned_resources == 2
    attack_response = game.attacker_best_response([(0, 0)], [1.0])
    assert 1 not in attack_response.allocation

    result = solve_double_oracle(game, max_iterations=10, tolerance=1e-9)
    assert all(1 not in allocation for allocation in result.defender_strategies)
    assert all(1 not in allocation for allocation in result.attacker_strategies)
    assert all(1 not in row.defender_best_response for row in result.history)
    assert all(1 not in row.attacker_best_response for row in result.history)


def test_count_support_mask_rejects_a_globally_infeasible_exact_budget():
    mask = np.asarray([[True, False, True]], dtype=bool)
    with pytest.raises(ValueError, match="no budget-feasible allocation"):
        EventBlottoGame(
            np.zeros((1, 3, 2), dtype=np.float64),
            defender_budget=1,
            attacker_budget=0,
            allow_defender_reserve=False,
            defender_count_mask=mask,
        )


def test_converged_iteration_does_not_claim_an_unapplied_oracle_addition():
    local = np.zeros((1, 2, 2), dtype=np.float64)
    local[0, 1, 0] = 0.75
    game = EventBlottoGame(local, defender_budget=1, attacker_budget=1)
    result = solve_double_oracle(
        game,
        initial_defender_strategies=[(0,)],
        initial_attacker_strategies=[(0,)],
        max_iterations=5,
        tolerance=1.0,
    )
    assert result.converged
    assert result.exploitability == pytest.approx(0.75)
    assert result.history[0].defender_gap > 0.5
    assert not result.history[0].added_defender_strategy
    assert not result.history[0].added_attacker_strategy
    assert result.defender_strategies == ((0,),)


def test_min_cost_matching_supports_demands_capacities_ids_and_reserve():
    costs = np.array(
        [
            [0.0, 8.0],
            [8.0, 0.0],
            [0.5, 3.0],
            [3.0, 3.0],
        ]
    )
    result = match_agents_to_tasks(
        costs,
        demands=[1, 1],
        capacities=[2, 1],
        reserve_cost=1.0,
        agent_ids=["r0", "r1", "r2", "r3"],
        task_ids=["left", "right"],
    )
    assert result.task_counts == (2, 1)
    assert result.task_agents == (("r0", "r2"), ("r1",))
    assert result.reserve_agents == ("r3",)
    assert result.total_cost == pytest.approx(1.5)
    assert result.assignment_by_agent() == {
        "r0": "left",
        "r1": "right",
        "r2": "left",
        "r3": None,
    }


def test_sparse_matching_reports_infeasible_demands():
    costs = np.array([[0.0, np.inf], [1.0, np.inf]])
    with pytest.raises(ValueError, match="infeasible"):
        match_agents_to_tasks(costs, demands=[1, 1], reserve_cost=0.0)


def test_coo_matching_matches_dense_api_with_ids_and_reserve():
    costs = np.array(
        [
            [0.0, 8.0],
            [8.0, 0.0],
            [0.5, 3.0],
            [3.0, 3.0],
        ]
    )
    dense = match_agents_to_tasks(
        costs,
        demands=[1, 1],
        capacities=[2, 1],
        reserve_cost=1.0,
        agent_ids=["r0", "r1", "r2", "r3"],
        task_ids=["left", "right"],
    )
    edge_agents, edge_tasks = np.nonzero(np.isfinite(costs))
    sparse = match_sparse_agents_to_tasks(
        n_agents=4,
        n_tasks=2,
        edge_agent_indices=edge_agents,
        edge_task_indices=edge_tasks,
        edge_costs=costs[edge_agents, edge_tasks],
        demands=[1, 1],
        capacities=[2, 1],
        reserve_cost=1.0,
        agent_ids=["r0", "r1", "r2", "r3"],
        task_ids=["left", "right"],
    )
    np.testing.assert_array_equal(sparse.assignment, dense.assignment)
    assert sparse.task_counts == dense.task_counts
    assert sparse.task_agents == dense.task_agents
    assert sparse.reserve_agents == dense.reserve_agents
    assert sparse.assignment_by_agent() == dense.assignment_by_agent()
    assert sparse.total_cost == pytest.approx(dense.total_cost)


def test_coo_matching_coalesces_duplicate_edges_to_the_cheapest_cost():
    result = match_sparse_agents_to_tasks(
        n_agents=3,
        n_tasks=2,
        edge_agent_indices=[0, 0, 1, 2],
        edge_task_indices=[0, 0, 1, 0],
        edge_costs=[9.0, 0.2, 0.1, 0.4],
        demands=[1, 1],
        capacities=[1, 1],
        reserve_cost=1.0,
        agent_ids=["a", "b", "c"],
        task_ids=["x", "y"],
    )
    assert result.assignment_by_agent() == {"a": "x", "b": "y", "c": None}
    assert result.total_cost == pytest.approx(1.3)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        (
            {
                "edge_agent_indices": [0, 1],
                "edge_task_indices": [0],
                "edge_costs": [0.1, 0.2],
            },
            ValueError,
            "equal length",
        ),
        (
            {
                "edge_agent_indices": [2],
                "edge_task_indices": [0],
                "edge_costs": [0.1],
            },
            ValueError,
            "out-of-range",
        ),
        (
            {
                "edge_agent_indices": [0],
                "edge_task_indices": [0],
                "edge_costs": [np.inf],
            },
            ValueError,
            "must be finite",
        ),
    ],
)
def test_coo_matching_validates_edge_lists(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        match_sparse_agents_to_tasks(
            n_agents=2,
            n_tasks=1,
            demands=[0],
            reserve_cost=0.0,
            **kwargs,
        )


def test_game_rejects_invalid_allocations_and_tensors():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        payoff_from_breach_risk(np.array([[[1.2]]]))
    game = _two_battlefield_protection_game()
    with pytest.raises(ValueError, match="resource sum"):
        game.validate_defender_allocation((0, 0))
    with pytest.raises(ValueError, match="capacity"):
        game.validate_attacker_allocation((2, 0))
