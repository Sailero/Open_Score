"""Research validity checks for public-history BA-DIB; no formal outcomes mocked."""
from dataclasses import replace
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from open_score.envs import HADStage3Adapter
from open_score.stage3.unknown_upper.analysis import paired_effect
from open_score.stage3.unknown_upper.belief import OpponentBelief
from open_score.stage3.unknown_upper.diagnostics import enumerate_actions,information_ladder
from open_score.stage3.unknown_upper.domain import IdentityUpperAction,PublicState,canonical_blue,match,observe,prune,seed_for
from open_score.stage3.unknown_upper.evaluation import formal_registry,training_registry
from open_score.stage3.unknown_upper.policies import decode_action,simple_blue_actions
from open_score.stage3.unknown_upper.qom import QuantizedOpponentModel,QOMRuntime,collate,loss
from open_score.stage3.unknown_upper.storage import digest,read_unit,write_unit
from open_score.stage3.unknown_upper.world import Execution,blue_destinations,from_public,isolated_rng,motion_log_likelihood,predicted_velocities

PROJECT=Path(__file__).resolve().parents[1]


@pytest.fixture
def state():
    adapter=HADStage3Adapter(4,4,2,target_positions=[[-2100.,-300.,100.],[-2100.,300.,100.]],blue_rule_style="split_rush")
    adapter.reset(seed=9876)
    return observe(adapter)


@pytest.mark.parametrize("red,blue,targets",[(18,12,2),(24,16,4),(30,20,5)])
def test_complete_id_cover_cap_and_reserves(red,blue,targets):
    from open_score.stage3.unknown_upper.domain import red_candidates
    adapter=HADStage3Adapter(red,blue,targets);adapter.reset(seed=red)
    state=observe(adapter)
    for action in red_candidates(state):
        action.validate(state.ids("red"),range(targets),blue_ids=state.ids("blue"))
        for style in ["balanced","concentrated_nearest","two_front","feint_switch"]:
            for b in simple_blue_actions(state,style)[0]:
                b.validate(state.ids("blue"),range(targets),"Blue")
                pairs=match(state,action,b)
                assert sorted(i for _,_,group in pairs for i in group)==sorted(state.ids("blue"))
                assert sorted([i for _,group,_ in pairs for i in group]+list(action.reserve_ids))==sorted(state.ids("red"))
                assert all(len(r)<=4 and len(b)<=4 for _,r,b in pairs)


def test_invalid_identity_and_intent_rejected(state):
    r=state.ids("red");b=state.ids("blue")
    with pytest.raises(ValueError):
        IdentityUpperAction(((0,r),(1,(r[0],)))).validate(r,[0,1])
    with pytest.raises(ValueError):
        IdentityUpperAction(((0,r),),(),((r,(999,)),)).validate(r,[0,1],blue_ids=b)
    with pytest.raises(ValueError):
        IdentityUpperAction(((0,b),),(),((b,(r[0],)),)).validate(b,[0,1],"Blue")


def test_rush_group_boundary_is_physically_invariant(state):
    state=replace(state,lower="rush")
    ids=state.ids("blue");r=IdentityUpperAction(((0,state.ids("red")),))
    first=IdentityUpperAction(((0,ids),))
    second=IdentityUpperAction(tuple((0,(i,)) for i in ids))
    assert canonical_blue(state,first)==canonical_blue(state,second)
    assert match(state,r,first)==match(state,r,second)
    for i in ids:
        np.testing.assert_allclose(predicted_velocities(state,first)[i],predicted_velocities(state,second)[i])


def test_split_boundaries_retain_real_motion_meaning(state):
    ids=state.ids("blue")
    first=IdentityUpperAction(((0,ids),));second=IdentityUpperAction(tuple((0,(i,)) for i in ids))
    assert any(not np.allclose(blue_destinations(state,first)[i],blue_destinations(state,second)[i]) for i in ids)


def test_explicit_interception_changes_pairing_without_channels(state):
    r,b=state.ids("red"),state.ids("blue")
    blue=IdentityUpperAction(((0,b[:2]),(0,b[2:])))
    plain=IdentityUpperAction(((0,r[:2]),(0,r[2:])))
    pairing=match(state,plain,blue)
    first_blue=next(bg for _,rg,bg in pairing if rg==r[:2])
    other=b[2:] if first_blue==b[:2] else b[:2]
    intent=IdentityUpperAction(plain.groups,(),((r[:2],other),))
    assert next(bg for _,rg,bg in match(state,intent,blue) if rg==r[:2])==other


def test_pruning_rebuilds_intents_after_casualties(state):
    r,b=state.ids("red"),state.ids("blue")
    action=IdentityUpperAction(((0,r),),(),((r,b),))
    survivor=prune(action,r[1:],b[1:])
    survivor.validate(r[1:],[0,1],blue_ids=b[1:])
    assert survivor.intercepts==((r[1:],b[1:]),)


def test_public_view_excludes_hidden_assignment_rng_and_type(state):
    adapter=from_public(state,1)
    before=observe(adapter)
    adapter.set_assignments("Blue",{i:1 for i in adapter.blue_ids})
    adapter.env.np_random=np.random.default_rng(987)
    assert observe(adapter)==before
    assert set(before.to_dict())=={"step","max_steps","lower","red","blue","targets","red_history_counts"}
    assert PublicState.from_dict(before.to_dict())==before
    clone=from_public(before,99)
    assert all(t is None for t in clone.blue_assignment.values())


def test_branch_rng_does_not_consume_official_future():
    np.random.seed(4321)
    expected=np.random.random(5)
    np.random.seed(4321)
    with isolated_rng(23):
        np.random.random(100)
    np.testing.assert_equal(np.random.random(5),expected)


def test_known_motion_likelihood_uses_new_public_transition(state):
    action=simple_blue_actions(state,"balanced")[0][0]
    velocity=predicted_velocities(state,action)
    after=replace(state,step=1,blue=tuple(replace(x,velocity=tuple(velocity[x.id])) for x in state.blue))
    assert motion_log_likelihood(state,after,action)==pytest.approx(0)
    belief=OpponentBelief(known_type="balanced")
    belief.prepare(state,None,np.random.default_rng(1))
    belief.update(state,after)
    with pytest.raises(ValueError,match="Repeated"):
        belief.update(state,after)
    assert belief.public_updates==1


def test_observable_equivalent_types_do_not_artificially_separate(state):
    action=simple_blue_actions(state,"balanced")[0][0]
    velocity=predicted_velocities(state,action)
    after=replace(state,step=1,blue=tuple(replace(x,velocity=tuple(velocity[x.id])) for x in state.blue))
    belief=OpponentBelief()
    belief.types=("a","b");belief.posterior=np.array([.5,.5]);belief.last_update_step=0
    belief.actions=[action,action];belief.labels=[0,1];belief.weights=np.array([.5,.5])
    belief.update(state,after)
    np.testing.assert_allclose(belief.posterior,[.5,.5])


def test_information_ladder_is_nested_but_counts_and_type_are_not():
    matrix=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.]])
    result=information_ladder(matrix,[0,0,1,1],[(0,),(1,),(0,),(1,)],[.25]*4)
    assert result["unknown"]==result["type"]==.5
    assert result["counts_only_separate_branch"]==result["full_action"]==1


def episode_for(state):
    action=simple_blue_actions(state,"balanced")[0][0]
    return {"commands":[{"public":state.to_dict(),"blue_action":action.to_dict()}]}


def test_qom_dynamic_inputs_and_labels_are_separate(state):
    torch.manual_seed(4)
    model=QuantizedOpponentModel()
    first=collate([episode_for(state)],"cpu")
    second={k:v.clone() for k,v in first.items()}
    second["target_labels"][:]=1
    with torch.no_grad():
        p1=model(first)[0];p2=model(second)[0]
    torch.testing.assert_close(p1,p2)
    value,_,_,_=loss(model,first)
    value.backward()
    assert torch.isfinite(value)
    assert model.temporal.weight_ih_l0.grad is not None


def test_qom_entity_permutation_equivariance(state):
    torch.manual_seed(11)
    model=QuantizedOpponentModel().eval()
    p,g=QOMRuntime(model).distributions(state)
    perm=[2,0,3,1]
    permuted=replace(state,blue=tuple(state.blue[i] for i in perm),red=tuple(reversed(state.red)))
    pp,gg=QOMRuntime(model).distributions(permuted)
    np.testing.assert_allclose(pp,p[:,perm,:],atol=1e-6)
    np.testing.assert_allclose(gg,g[:,perm,:][:,:,perm],atol=1e-6)


def test_qom_constrained_decode_large_rosters():
    adapter=HADStage3Adapter(30,20,5);adapter.reset(seed=2);state=observe(adapter)
    for seed in range(8):
        p=np.ones((20,5))/5;g=np.ones((20,20))
        action=decode_action(state,p,g,np.random.default_rng(seed))
        action.validate(state.ids("blue"),range(5),"Blue")


def test_formal_registry_complete_paired_and_disjoint():
    config=yaml.safe_load((PROJECT/"configs/stage123_unknown_upper.yaml").read_text(encoding="utf8"))
    train,formal=training_registry(config),formal_registry(config)
    assert len(train)==2400 and len(formal)==1920
    assert [sum(c["split"]==s for c in train) for s in ["train","validation","test"]]==[1680,360,360]
    assert not {c["seed"] for c in train}&{c["seed"] for c in formal}
    assert all(c["upper"]!="feint_switch" for c in train)
    groups={}
    for c in formal:groups.setdefault(c["seed"],set()).add(c["method"])
    assert len(groups)==240 and all(len(g)==8 for g in groups.values())
    assert seed_for("blue",1,5)!=seed_for("red-planner",1,5)


def test_resume_rejects_corrupt_or_changed_units(tmp_path):
    path=tmp_path/"cell.json.gz";result={"win":1,"ids":[2,3]}
    write_unit(path,"input",result)
    assert read_unit(path,"input")==result
    assert read_unit(path,"changed") is None
    path.write_bytes(path.read_bytes()[:20])
    assert read_unit(path,"input") is None


def test_compressed_training_evidence_does_not_encode_worker_filename(tmp_path):
    first,second=tmp_path/"first.json.gz",tmp_path/"second.json.gz"
    for path in [first,second]:
        write_unit(path,"same_input",{"same_episode":[1,2,3]})
    assert first.read_bytes()==second.read_bytes()


def test_small_complete_domain_contains_reserve_and_interception(state):
    r,b=state.ids("red")[:2],state.ids("blue")[:2]
    actions=enumerate_actions(r,[0],True,b)
    assert IdentityUpperAction((),r) in actions
    assert any(a.intercepts for a in actions)
    for a in actions:a.validate(r,[0],blue_ids=b)


def test_paired_statistics_never_pair_different_seeds():
    rows=[]
    for i in range(10):
        for method,win in [("a",1),("b",0)]:
            rows.append({"cell":{"seed":i,"method":method,"scenario":{"label":"small"},"lower":"rush","upper":"balanced"},"red_win":win})
    effect=paired_effect(rows,"a","b",100)
    assert effect["pairs"]==10 and effect["difference"]==1 and effect["ci95"]==[1,1]
    rows[-1]["cell"]["seed"]=999
    assert paired_effect(rows,"a","b",100)["pairs"]==9


def test_frozen_joint_interface_preserves_calibrated_marginal(tmp_path,state,monkeypatch):
    from open_score.stage2 import DynamicHADOutcomeNet,HADCanonicalizer
    from open_score.stage3.payoff import FrozenStage2Payoff
    model=DynamicHADOutcomeNet(10,16,32)
    path=tmp_path/"stage2.pt"
    torch.save({"model":model.state_dict(),"horizon_bins":10,"steps_per_bin":5,"entity_hidden_dim":16,"hidden_dim":32,"temperature":1.2,
                "style_calibration":{"offsets":{"rush|4|4":.3},"global_offsets":{"rush":.1},"source_split":"calibration"}},path)
    predictor=FrozenStage2Payoff(path)
    adapter=from_public(state)
    local=HADCanonicalizer().to_entity_set(adapter.local_state_entities(0,state.ids("red"),state.ids("blue")))
    joint=predictor.predict_joint_outcome_time([local],styles=["rush"])
    assert joint.shape==(1,2,10)
    np.testing.assert_allclose(joint.sum((1,2)),1.,atol=1e-6)
    np.testing.assert_allclose(joint[:,0,:].sum(-1),predictor.predict_red_win([local],styles=["rush"]),atol=1e-6)
    assert predictor.predict_joint_outcome_time([]).shape==(0,2,10)
    logits=torch.full((1,20),-1000.0)
    logits[0,0]=1000.0
    logits[0,15]=0.0
    monkeypatch.setattr(predictor.model,"forward",lambda target,*args:logits.expand(len(target),-1))
    extreme=predictor.predict_joint_outcome_time([local],styles=["rush"])
    np.testing.assert_allclose(extreme.sum(),1.,atol=1e-7)
    np.testing.assert_allclose(extreme[:,0].sum(-1),predictor.predict_red_win([local],styles=["rush"]),atol=1e-6)
    assert extreme[0,1,5]>0 and np.count_nonzero(extreme[0,1])==1


def test_next_macro_boundary_respects_global_clock(monkeypatch,state):
    from open_score.stage3.unknown_upper import world
    start=replace(state,step=12)
    class Adapter:
        step_count=12
        def _terminal_sign(self):return 0
    adapter=Adapter()
    class FakeExecution:
        def __init__(self,*args):pass
        def step(self,a):
            a.step_count+=1
            return None,None,False,{"events":[]}
    monkeypatch.setattr(world,"from_public",lambda *a:adapter)
    monkeypatch.setattr(world,"Execution",FakeExecution)
    monkeypatch.setattr(world,"observe",lambda a,*args:replace(start,step=a.step_count))
    end,outcome,trace=world.branch(start,None,None,None,None,3)
    assert end.step==15 and len(trace)==4 and outcome==0


def test_candidate_domain_is_independent_of_belief(state):
    from open_score.stage3.unknown_upper.policies import shared_candidates
    class Proxy:
        def values(self,reds,blues,final=False):
            return np.asarray([[-float(len(r.reserve_ids)) for b in blues] for r in reds])
    first=shared_candidates(state,Proxy())[0]
    second=shared_candidates(PublicState.from_dict(state.to_dict()),Proxy())[0]
    assert first==second
    for action in first:action.validate(state.ids("red"),[0,1],blue_ids=state.ids("blue"))


def test_current_filter_retains_within_code_action_evidence(state):
    from types import SimpleNamespace
    qom = SimpleNamespace(model=SimpleNamespace(codes=1), prior=np.ones(1))
    belief = OpponentBelief("qom", qom)
    ids = state.ids("blue")
    belief.actions = [IdentityUpperAction(((t,ids),)) for t in [0,1]]
    belief.labels = [0,0]
    belief.weights = np.array([.9,.1])
    belief.event_distributions = (np.full((1,len(ids),2),.5),None)
    np.testing.assert_allclose(belief.targets(state)[:,0],.5)
    np.testing.assert_allclose(belief.current_targets(state)[:,0],.9)


def test_filter_and_unseen_next_action_are_different_metrics():
    from open_score.stage3.unknown_upper.analysis import inference_metrics
    row = {"cell":{"method":"qom_mcp","upper":"balanced"},
           "events":[{"event_index":1,"truth_targets":[1],"predicted_targets":[0],"forecast_oracle_accuracy_ceiling":.5}],
           "filtered_assignments":[{"event_index":1,"truth_targets":[1],"predicted_targets":[1]}],
           "surprise":[{"step":11,"nll":0.0}]}
    result = inference_metrics([row])
    assert result["target_allocation_micro_f1"] == 1.0
    assert result["next_action_forecast_micro_f1"] == 0.0
    assert result["same_state_known_program_forecast_accuracy_ceiling"] == .5
