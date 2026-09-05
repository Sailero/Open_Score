"""Fast path must preserve neutral matching, physics, and paired conditions."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import yaml
from open_score.envs import HADStage3Adapter
from open_score.stage3.unknown_upper.domain import observe, match, reference_match, red_candidates
from open_score.stage3.unknown_upper.policies import simple_blue_actions
from open_score.stage3.unknown_upper.fast_evaluation import registry
from open_score.stage3.unknown_upper.fast_planner import METHODS
from open_score.stage3.unknown_upper.world import from_public, isolated_rng


@pytest.mark.parametrize('red,blue,targets', [(18,12,2),(24,16,4),(30,20,5)])
@pytest.mark.parametrize('lower', ['rush','split_rush'])
def test_cached_geometry_preserves_reference_matching(red, blue, targets, lower):
    adapter = HADStage3Adapter(red,blue,targets,blue_rule_style=lower)
    adapter.reset(seed=773)
    state = observe(adapter)
    for r in red_candidates(state)[:4]:
        for kind in ('balanced','two_front'):
            for b in simple_blue_actions(state,kind)[0][::2]:
                assert match(state,r,b) == reference_match(state,r,b)
                assert match(state,r,b) == reference_match(state,r,b)


@pytest.mark.parametrize('lower',['rush','split_rush'])
def test_physics_only_step_matches_legacy_full_step(lower):
    source = HADStage3Adapter(6,4,2,blue_rule_style=lower)
    source.reset(seed=521)
    fast, old = from_public(observe(source),42), from_public(observe(source),42)
    old.env.step_physics = old.env.step
    for adapter in (fast,old):
        adapter.set_joint_assignments({i:i%2 for i in adapter.red_ids}, {i:i%2 for i in adapter.blue_ids})
    for step in range(3):
        results=[]
        for adapter in (fast,old):
            with isolated_rng(88+step):
                _,reward,done,info=adapter.step({i:0 for i in adapter.red_ids})
            results.append((observe(adapter),reward,done,info['events']))
        assert results[0] == results[1]
        if results[0][2]:
            break


def test_all_six_methods_share_every_registered_condition_and_seed():
    cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/stage3_fast_compare.yaml').read_text(encoding='utf-8'))
    cells=registry(cfg)
    assert len(cells)==720 and len({c['id'] for c in cells})==720
    by_seed={}
    for c in cells:
        by_seed.setdefault(c['seed'],[]).append(c)
    assert len(by_seed)==120
    for rows in by_seed.values():
        assert {r['method'] for r in rows}==set(METHODS)
        assert len({(r['scenario']['label'],r['lower'],r['upper'],r['repeat']) for r in rows})==1


def test_runner_registry_and_content_hash_resume_agree(tmp_path):
    import importlib.util
    from open_score.stage3.unknown_upper.storage import write_unit
    script=Path(__file__).parents[1]/'scripts/run_stage3_fast_compare.py'
    spec=importlib.util.spec_from_file_location('fast_runner_for_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=yaml.safe_load((script.parents[1]/'configs/stage3_fast_compare.yaml').read_text(encoding='utf-8'))
    assert module.register(cfg)==registry(cfg)
    path=tmp_path/'unit.json.gz'
    result={'cell':module.register(cfg)[0],'red_win':1}
    write_unit(path,'hash_v2',result)
    assert module.read_unit(path,'hash_v2')==result
    assert module.read_unit(path,'changed') is None
    path.write_bytes(b'corrupted')
    assert module.read_unit(path,'hash_v2') is None


def test_summary_preserves_pairing_and_rejects_changed_conditions(tmp_path):
    from open_score.stage3.unknown_upper.fast_analysis import summarize
    cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/stage3_fast_compare.yaml').read_text(encoding='utf-8'))
    cfg['acceptance']['bootstrap_replicates']=100
    rows=[{'cell':c,'red_win':int(c['method']!='balanced'),'events':[{'planning_seconds':.1}],
           'filtered_assignments':[],'wall_seconds':1.,'steps':5,'identity_valid':True,
           'fixed_targets':True,'infeasible':False} for c in registry(cfg)]
    result=summarize(rows,cfg,tmp_path)
    assert result['paired_effects']['belief_short:balanced']['difference']==1.
    assert result['paired_effects']['belief_short:known_short']['difference']==0.
    assert result['pairs_per_method']==120
    assert not result['acceptance']['belief_short']['significant_gain']
    assert (tmp_path/'episodes.csv').exists()
    rows[0]['cell']={**rows[0]['cell'],'lower':'changed'}
    with pytest.raises(ValueError):
        summarize(rows,cfg,tmp_path)


def test_no_red_command_still_prepares_public_filter(monkeypatch):
    import open_score.stage3.unknown_upper.fast_planner as planner
    from open_score.stage3.unknown_upper.belief import OpponentBelief
    adapter=HADStage3Adapter(2,2,2);adapter.reset(seed=2)
    state=observe(adapter)
    state=replace(state,red=tuple(replace(r,health=0) for r in state.red))
    belief=OpponentBelief(known_type='balanced')
    monkeypatch.setattr(planner,'RiskProxy',lambda *args:None)
    commander=planner.FastCommander(None,None,None,{'decision_seconds':2.5})
    action,_=commander.plan(state,'known_short',belief,np.random.default_rng(1))
    assert not action.groups and belief.actions and belief.last_update_step==state.step


def test_fixed_public_candidate_domain_has_no_type_or_action_input():
    import inspect
    from open_score.stage3.unknown_upper.fast_planner import candidates, continuation_candidates
    assert list(inspect.signature(candidates).parameters)==['state','previous']
    assert list(inspect.signature(continuation_candidates).parameters)==['state','previous','belief']
    adapter=HADStage3Adapter(6,4,2);adapter.reset(seed=22)
    state=observe(adapter)
    pool=candidates(state)
    for action in pool:
        action.validate(state.ids('red'),[t.id for t in state.targets],blue_ids=state.ids('blue'))
    assert pool==candidates(state)
