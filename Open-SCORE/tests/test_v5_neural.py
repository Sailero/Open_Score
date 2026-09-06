"""Probabilities, variable rosters and real optimization of the new V5 models."""
from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch

from open_score.grouping.domain import Group,Grouping
from open_score.research_v4.actions import neighbors,partition_key,rule_grouping
from open_score.research_v4.environment import make_env
from open_score.research_v5.planning import propose_plans
from open_score.research_v5.neural import (AutoregressiveActor,CandidateActor,StateCritic,
    NeuralPolicy,CoverageProvider,plan_trace,event_gae,ppo_update,imitation_update,actor_evaluate)


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)
    torch.manual_seed(921)


def test_constructive_probabilities_sum_over_complete_small_domain():
    state=make_env(3,2).reset(52)
    start=Grouping((),state.ids('red'))
    found,frontier={partition_key(start):start},[start]
    while frontier:
        for plan in neighbors(state,frontier.pop()):
            key=partition_key(plan)
            if key not in found:
                found[key]=plan
                frontier.append(plan)
    plans=list(found.values())
    assert len(plans)==47
    traces=[plan_trace(state,p) for p in plans]
    assert len(set(map(tuple,traces)))==47
    actor=AutoregressiveActor(32,4,1)
    result=actor([state]*len(plans),traces)
    assert result['plans']==plans
    torch.testing.assert_close(result['log_prob'].exp().sum(),torch.tensor(1.),atol=2e-6,rtol=2e-6)


def test_mixed_rosters_full_deployment_reserve_and_trace_replay():
    states=[make_env(r,b).reset(812) for r,b in ((8,4),(8,6),(12,9),(32,16))]
    actor=AutoregressiveActor(32,4,1)
    plans=[Grouping((Group(s.targets[0].id,s.ids('red')),)) for s in states]
    traces=[plan_trace(s,p) for s,p in zip(states,plans)]
    result=actor(states,traces)
    assert result['plans']==plans
    assert len(result['plans'][-1].groups[0].members)==32
    sampled=actor(states)
    replay=actor(states,sampled['traces'])
    torch.testing.assert_close(sampled['log_prob'],replay['log_prob'])
    torch.testing.assert_close(sampled['joint_entropy'],sampled['entropy']*torch.tensor([8,8,12,32]))
    reserved=actor([states[0]],[plan_trace(states[0],Grouping((),states[0].ids('red')))])
    assert not reserved['plans'][0].groups
    with pytest.raises(ValueError):
        actor([states[0]],[[99]*8])


def test_candidate_entity_input_order_invariance_and_membership_response():
    state=make_env(8,6).reset(27)
    reverse=replace(state,red=tuple(reversed(state.red)),blue=tuple(reversed(state.blue)),targets=tuple(reversed(state.targets)))
    pool=propose_plans(state,8)
    actor=CandidateActor(32,4,1)
    scores=actor([state,reverse],[pool,pool])
    torch.testing.assert_close(scores[0],scores[1],atol=1e-6,rtol=1e-5)
    assert float((scores[0].max()-scores[0].min()).detach())>1e-6


def test_coverage_reproducible_does_not_consume_policy_rng_and_is_picklable():
    state=make_env(8,4).reset(9)
    provider=CoverageProvider(NeuralPolicy(AutoregressiveActor(32,4,1),'autoregressive'),8,11)
    before=torch.get_rng_state().clone()
    first=provider(state)
    assert torch.equal(before,torch.get_rng_state())
    assert first==provider(state)
    assert first==pickle.loads(pickle.dumps(provider))(state)
    assert len(first)==8


def test_gae_true_terminal_and_batch_bootstrap():
    rows=[dict(reward=0.,done=False,value=.2,env=0),dict(reward=1.,done=True,value=.4,env=0),
          dict(reward=0.,done=False,value=.1,env=0)]
    advantages,returns=event_gae(rows,[.4,999.,.7],.95)
    np.testing.assert_allclose(advantages,[.2+.95*.6,.6,.6])
    np.testing.assert_allclose(returns,[.97,1.,.7])


def test_candidate_uniform_entropy_is_log_candidates_independent_of_red_count_and_used_by_ppo():
    states=[make_env(red,red//2).reset(91) for red in (8,32)]
    actor=CandidateActor(32,4,1)
    critic=StateCritic(32,4,1)
    with torch.no_grad():
        for model in (actor,critic):
            for parameter in model.parameters(): parameter.zero_()
    rows=[]
    for state in states:
        pool=propose_plans(state,4)
        assert len(pool)==4
        rows.append(dict(state=state,next_state=state,done=True,reward=0.,value=0.,
            log_prob=-float(np.log(4)),action=0,candidates=pool))
    logp,regularizer,joint=actor_evaluate(actor,'candidate',rows)
    expected=torch.full((2,),float(np.log(4)))
    torch.testing.assert_close(regularizer,expected)
    torch.testing.assert_close(joint,expected)
    torch.testing.assert_close(logp,-expected)
    ao=torch.optim.SGD(actor.parameters(),lr=.01)
    co=torch.optim.SGD(critic.parameters(),lr=.01)
    metrics=ppo_update(actor,critic,ao,co,rows,'candidate',dict(batch_size=2,epochs=1,
        entropy_start=.02,entropy_end=.02))
    assert metrics['actor_updates']==1 and metrics['actor_loss']==0
    assert metrics['entropy']==pytest.approx(np.log(4))
    assert metrics['actor_total_loss']==pytest.approx(-.02*np.log(4))
    assert metrics['per_member_entropy']==pytest.approx(np.log(4)*(1/8+1/32)/2)


@pytest.mark.parametrize('kind',['candidate','autoregressive'])
def test_bc_and_ppo_actual_independent_parameter_updates(kind):
    state=make_env(4,2).reset(87)
    actor=(CandidateActor if kind=='candidate' else AutoregressiveActor)(32,4,1)
    critic=StateCritic(32,4,1)
    assert not set(map(id,actor.parameters()))&set(map(id,critic.parameters()))
    ao=torch.optim.Adam(actor.parameters(),lr=3e-4)
    co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    pool=propose_plans(state,4)
    original=torch.cat([p.detach().flatten() for p in actor.parameters()]).clone()
    metric=imitation_update(actor,ao,[dict(state=state,teacher=rule_grouping(state),candidates=pool)],kind)
    assert metric['optimizer_steps']==1
    assert not torch.equal(original,torch.cat([p.detach().flatten() for p in actor.parameters()]))
    rows=[]
    for reward in (0.,1.,0.,1.):
        with torch.no_grad():
            value=float(critic([state])[0])
            if kind=='candidate':
                dist=torch.distributions.Categorical(logits=actor([state],[pool])[0])
                selected=int(dist.sample())
                logp=float(dist.log_prob(torch.tensor(selected)))
                extra=dict(action=selected,candidates=pool)
            else:
                decision=actor([state])
                logp=float(decision['log_prob'][0]);extra=dict(trace=decision['traces'][0])
        rows.append(dict(state=state,next_state=replace(state,red=(),blue=(),targets=()),done=True,
            reward=reward,value=value,log_prob=logp,env=0,**extra))
    metric=ppo_update(actor,critic,ao,co,rows,kind,dict(batch_size=4,epochs=1))
    assert metric['actor_updates']==metric['critic_updates']==1
    assert metric['approx_kl']<1e-5
    assert metric['actor_total_loss']==pytest.approx(metric['actor_loss']-.02*metric['entropy'],abs=1e-6)
    assert metric['entropy']==pytest.approx(metric['joint_entropy'] if kind=='candidate' else metric['per_member_entropy'])
