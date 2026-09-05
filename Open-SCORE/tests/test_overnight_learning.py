"""Independent arithmetic and real-gradient checks for overnight learning."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from open_score.grouping.domain import DecisionState, Entity, Grouping
from open_score.overnight.actions import candidate_actions
from open_score.overnight.config import configuration
from open_score.overnight.learning import vector_gae, double_q_update, ppo_update
from open_score.overnight.policy import CandidateNetwork
from open_score.overnight.rewards import potential, shaped_reward


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(127)
    yield
    torch.set_num_threads(previous)


def make_state(size=4, step=0):
    red = tuple(Entity(i, (-600 + 170*i, 350 - 100*i, 100), (70, 0, 0), 1) for i in range(size))
    blue = tuple(Entity(100+i, (1700 - 100*i, -200 + 70*i, 100), (-130, 0, 0), 1) for i in range(size))
    targets = (Entity(0, (-2100, -600, 100), (0, 0, 0), 1.2), Entity(1, (-2100, 600, 100), (0, 0, 0), 1.2))
    return DecisionState(step, 50, 'reactive', red, blue, targets, Grouping((), tuple(range(size))),
                         {i: (0.,)*64 for i in range(size)}, {i: -1 for i in range(size)})


def test_gae_interleaved_streams_match_hand_calculation_and_cut_at_real_terminal():
    # env 0: residuals .5 and 1; A[0]=.5+.5*1=1.
    # env 1: terminal residual -.5. Its later new episode has residual
    # 2+bootstrap(4)-value(3)=3 and must not flow through the old terminal.
    rows = [dict(env=0, reward=0., value=1., done=False),
            dict(env=1, reward=0., value=.5, done=True),
            dict(env=0, reward=3., value=2., done=True),
            dict(env=1, reward=2., value=3., done=False)]
    advantages, returns = vector_gae(rows, np.array([1.5, 999., 999., 4.]), lam=.5)
    np.testing.assert_allclose(advantages, [1., -.5, 1., 3.])
    np.testing.assert_allclose(returns, [2., 0., 3., 6.])
    # Positive terminal bootstrap sentinels must be entirely ignored.
    changed, _ = vector_gae(rows, np.array([1.5, -999., -999., 4.]), lam=.5)
    np.testing.assert_allclose(changed, advantages)


@pytest.mark.parametrize('success', [0., 1.])
def test_potential_telescopes_at_true_50_step_terminal_including_loss(success):
    initial = make_state()
    middle = replace(initial, step=20, blue=tuple(replace(entity, position=(-200., entity.position[1], 100.))
                                                   for entity in initial.blue))
    terminal = replace(middle, step=50,
                       blue=(replace(middle.blue[0], health=0.),) + middle.blue[1:])
    rewards = [shaped_reward(initial, middle, 0., False),
               shaped_reward(middle, terminal, success, True)]
    assert sum(rewards) == pytest.approx(success - potential(initial), abs=1e-12)
    assert potential(terminal, terminal=True) == 0.
    # A mere rollout boundary keeps its potential and requires bootstrap;
    # prematurely zeroing it would create a different training return.
    assert rewards[0] == pytest.approx(potential(middle) - potential(initial))
    assert potential(middle) != 0.


def test_potential_telescopes_at_early_termination_too():
    initial = make_state()
    early = replace(initial, step=7, red=tuple(replace(entity, health=0.) for entity in initial.red))
    assert shaped_reward(initial, early, 0., True) == pytest.approx(-potential(initial))


class TableNetwork(nn.Module):
    """Analytic two-state Q table, independent of the coalition implementation."""
    def __init__(self, table):
        super().__init__()
        self.table = nn.Parameter(torch.tensor(table, dtype=torch.float32))

    def forward(self, states, pools):
        scores = self.table[torch.tensor(states)]
        return scores, scores.sum(1)*0.


def test_double_q_uses_online_choice_target_value_and_never_bootstraps_terminal():
    online = TableNetwork([[1., 2.], [4., 3.]])
    target = TableNetwork([[0., 0.], [10., 100.]]).requires_grad_(False)
    optimizer = torch.optim.SGD(online.parameters(), lr=.1)
    rows = [dict(state=0, candidates=[0, 1], action=1, reward=1., done=False,
                 next_state=1, next_candidates=[0, 1]),
            dict(state=0, candidates=[0, 1], action=0, reward=-2., done=True,
                 next_state=1, next_candidates=[0, 1])]
    metrics = double_q_update(online, target, optimizer, rows, dict(max_gradient_norm=10.))
    # Online selects next action 0, although target prefers action 1.
    # Bellman targets are 11 and -2; Q estimates before update are 2 and 1.
    assert metrics['target_mean'] == pytest.approx(4.5)
    assert metrics['q_mean'] == pytest.approx(1.5)
    assert metrics['loss'] == pytest.approx(5.5)
    torch.testing.assert_close(online.table[0], torch.tensor([.95, 2.05]))
    torch.testing.assert_close(online.table[1], torch.tensor([4., 3.]))
    assert all(parameter.grad is None for parameter in target.parameters())


def test_real_candidate_ppo_updates_encoder_and_preserves_padding_masks():
    states = [make_state(4), make_state(8)]
    pools = [candidate_actions(states[0], 3), candidate_actions(states[1], 8)]
    network = CandidateNetwork(hidden_dim=32, heads=4, layers=1)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    with torch.no_grad():
        old_scores, old_values = network(states, pools)
        old_log = torch.distributions.Categorical(logits=old_scores).log_prob(torch.tensor([0, 1]))
    before_encoder = network.encoder.layers[0].self_attn.in_proj_weight.detach().clone()
    rows = [dict(state=states[i], candidates=pools[i], env=i, action=i,
                 value=float(old_values[i]), log_prob=float(old_log[i]),
                 reward=float(i), done=True, next_state=replace(states[i], step=50),
                 next_candidates=pools[i], delta=5) for i in range(2)]
    config = configuration('ppo_structured', smoke=True)
    metrics = ppo_update(network, optimizer, rows, config, fraction=.2)
    assert metrics['optimizer_steps'] >= 1 and np.isfinite(metrics['loss'])
    assert metrics['gradient_norm'] > 0
    assert not torch.equal(before_encoder, network.encoder.layers[0].self_attn.in_proj_weight)
    assert all(torch.isfinite(parameter.grad).all() for parameter in network.parameters() if parameter.grad is not None)
    scores, values = network(states, pools)
    assert torch.isfinite(values).all()
    assert torch.all(scores[0, len(pools[0]):] == -1e9)
    for i, pool in enumerate(pools):
        index = int(scores[i].argmax())
        assert index < len(pool)
        pool[index].validate(states[i].ids('red'), states[i].ids('targets'))
