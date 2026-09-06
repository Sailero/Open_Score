"""Semantic checks for the simulator planner, including stochastic DPW."""
import copy
import pickle

import numpy as np
import pytest

from open_score.research_v4.environment import make_env
from open_score.research_v4.actions import rule_grouping, current_grouping
from open_score.research_v5.planning import (
    DPWSearch, MCTSPolicy, RolloutPolicy, choose_with_rule_ties,
    propose_plans, terminal_rollouts,
)


class TwoStageMDP:
    """Exact optimal value 0.8; a cached chance outcome must stay stochastic."""
    @staticmethod
    def state_key(state):
        return state

    @staticmethod
    def action_key(action):
        return action

    @staticmethod
    def serialize_action(action):
        return action

    @staticmethod
    def propose(state, index, rng):
        return index % 2

    @staticmethod
    def tie_rank(state, action):
        return int(action == 0)

    @staticmethod
    def transition(state, action, seed):
        rng = np.random.default_rng(seed)
        if state[0] == 0:
            # Both chance branches have the same optimal second-stage action;
            # 0.5 branch generation is deliberately unrelated to tree visits.
            branch = int(rng.random() < .5)
            return (1, action, branch), 0., False, 1
        probability = .8 if action == 1 else .2
        if state[1] == 0:
            probability = .4
        reward = int(rng.random() < probability)
        return (2, reward), float(reward), True, 1

    @classmethod
    def rollout(cls, state, seed):
        _, reward, _, steps = cls.transition(state, 0, seed)
        return reward, steps


def test_dpw_discovers_future_choice_and_empirical_chance_frequencies():
    search = DPWSearch(TwoStageMDP(), iterations=8000, depth=2, seed=13)
    action, trace = search.search((0,))
    assert action == 1
    chosen = next(row for row in trace['root_actions'] if row['action'] == 1)
    assert .72 < chosen['value'] < .85
    assert chosen['successors'] == 2
    assert trace['actual_max_depth'] == 2
    root_draws = [r for r in trace['simulation_draws'] if r['kind'] == 'transition' and r['depth'] == 1]
    branches = [int(r['successor_key'].endswith('1)')) for r in root_draws]
    assert .46 < np.mean(branches) < .54
    assert trace['online_planner_steps'] == trace['simulation_physical_steps']


def test_depth_one_cannot_claim_multi_event_lookahead():
    _, trace = DPWSearch(TwoStageMDP(), iterations=2000, depth=1, seed=14).search((0,))
    assert trace['actual_max_depth'] == 1
    assert max(row['value'] for row in trace['root_actions']) < .5


def test_physical_budget_finishes_simulation_before_stopping():
    _, trace = DPWSearch(TwoStageMDP(), iterations=2000, depth=1, seed=14,
                         physical_budget=31).search((0,))
    assert 31 <= trace['simulation_physical_steps'] <= 32
    assert trace['iterations'] < trace['iteration_limit']
    assert trace['physical_budget_excess'] <= 1


def test_rule_is_exact_tie_default_even_if_current_was_first():
    env = make_env(4, 2, seed=31)
    state = env.reset()
    plans = propose_plans(state, 8, 7)
    assert plans[0] == current_grouping(state)
    index = choose_with_rule_ties(state, plans, [0.] * len(plans))
    assert plans[index] == rule_grouping(state)
    assert len(set(plans)) == len(plans)


def test_rollout_restores_physical_state_and_future_rng_exactly():
    env = make_env(4, 2, seed=31)
    state = env.reset()
    before = pickle.dumps(env.snapshot())
    ambient = pickle.dumps(np.random.get_state())
    plans = propose_plans(state, 4, 7)
    rows = terminal_rollouts(env, plans, [17, 19])
    assert pickle.dumps(env.snapshot()) == before
    assert pickle.dumps(np.random.get_state()) == ambient
    assert all(len(row['branches']) == 2 for row in rows)
    assert all(branch['physical_steps'] > 0 for row in rows for branch in row['branches'])
    assert all(row['value'] == np.mean(row['outcomes']) for row in rows)


def test_simulator_restore_survives_invalid_candidate():
    env = make_env(4, 2, seed=31)
    env.reset()
    before = pickle.dumps(env.snapshot())
    from open_score.grouping.domain import Grouping
    with pytest.raises(ValueError):
        terminal_rollouts(env, [Grouping(())], [10])
    assert pickle.dumps(env.snapshot()) == before


@pytest.mark.parametrize('policy', [RolloutPolicy(candidates=4, branches=2), MCTSPolicy(iterations=8, depth=2)])
def test_real_planners_never_mutate_the_live_episode(policy):
    env = make_env(4, 2, seed=32)
    state = env.reset()
    before = pickle.dumps(env.snapshot())
    plan = policy.act_env(env)
    plan.validate(state.ids('red'), state.ids('targets'), max_members=None)
    assert pickle.dumps(env.snapshot()) == before
    assert policy.last_trace['online_planner_steps'] > 0
    assert plan == policy.act_env(env)


def test_branch_seeds_are_separate_from_real_rng():
    env = make_env(4, 2, seed=33)
    env.reset()
    policy = RolloutPolicy(candidates=4, branches=2, seed=8)
    first = policy.act_env(env)
    trace = copy.deepcopy(policy.last_trace['candidates'])
    env.set_rng(987654)
    assert first == policy.act_env(env)
    assert trace == policy.last_trace['candidates']


@pytest.mark.parametrize('task_id', ['T1', 'T2'])
def test_task_entrypoint_completes_and_reuses_committed_results(tmp_path, task_id):
    from open_score.grouping.storage import atomic_json
    from open_score.research_v5.protocol import defaults, episode_spec
    from open_score.research_v5.runtime import TaskContext
    from open_score.research_v5.tasks import t1_rollout, t2_mcts
    config = defaults(smoke=True)
    config.update(cells=[[4, 2]], diagnostic_states=1, own_diagnostic_states=1,
                  selection_branches=1, verification_branches=1, device='cpu',
                  frozen_v4_assets=str(tmp_path/'missing_historical'))
    config['t1'].update(candidates=3, branches=1, control_episodes=1)
    config['t2'].update(iterations=3, depth=2)
    shared = tmp_path/'shared'
    shared.mkdir()
    for split, filename in [('test', 'evaluation_manifest.json'), ('validation', 'validation_manifest.json')]:
        spec = episode_spec(config['seed'], 0, split, 'shared', config['cells'])
        atomic_json(shared/filename, {'episodes': [spec.to_dict()]})
    context = TaskContext(tmp_path, task_id, config)
    execute = t1_rollout.run if task_id == 'T1' else t2_mcts.run
    first = execute(context)
    assert first['execution_status'] == 'completed'
    assert first['main']['complete']
    assert first['main']['episodes'] == 1
    episode_files = sorted(context.output.glob('evaluations/**/episodes.jsonl'))
    original = {str(path): path.read_bytes() for path in episode_files}
    execute(context)
    assert original == {str(path): path.read_bytes() for path in episode_files}
    assert (context.output/'phase_costs.jsonl').is_file()
