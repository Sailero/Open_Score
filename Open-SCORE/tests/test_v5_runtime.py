import gzip
import json
import pickle
import os
import time
import multiprocessing as mp

import numpy as np
import pytest

from open_score.research_v4.actions import rule_grouping, grand_grouping
from open_score.research_v4.policies import RulePolicy
from open_score.research_v4.runner import atomic_json
from open_score.research_v5.protocol import CELLS, defaults, digest, episode_spec
from open_score.research_v5.runtime import TaskContext
from open_score.research_v5.simulator import make_env, paired_rollouts
from open_score.research_v5.reporting import paired_interval


def context(tmp_path):
    config=defaults(smoke=True)
    config.update(cells=[[8,4]],bc_episodes=0,diagnostic_states=0,device='cpu')
    shared=tmp_path/'shared';shared.mkdir(exist_ok=True)
    atomic_json(shared/'config_resolved.json',config)
    atomic_json(shared/'protocol_resolved.json',{'protocol_hash':digest(config)})
    for split,name in [('test','evaluation'),('validation','validation')]:
        spec=episode_spec(config['seed'],0,split,'shared',config['cells'])
        atomic_json(shared/f'{name}_manifest.json',{'episodes':[spec.to_dict()]})
    return TaskContext(tmp_path,'T1',config)


def test_mixed_counts_all_fifteen_native_terminals_and_family_isolation():
    families=set()
    for index,(red,blue) in enumerate(CELLS):
        spec=episode_spec(20260907,index,'test')
        assert (spec.red_count,spec.blue_count)==(red,blue)
        assert spec.family_id not in families
        assert spec.family_id!=episode_spec(20260907,index,'train').family_id
        families.add(spec.family_id)
        env=make_env(spec)
        while not env.done:
            before=env.state().step
            after,reward,done,info=env.step(rule_grouping(env.state()))
            assert after.step-before==info['delta']>0
            assert reward==float(done and info['success'])
        assert 1<=env.state().step<=50


def test_private_branch_rng_and_candidate_order_leave_live_state_unchanged():
    env=make_env(episode_spec(20260907,0,'diagnostic'))
    snapshot=env.snapshot();before=pickle.dumps(snapshot)
    ambient=pickle.dumps(np.random.get_state())
    plans=[rule_grouping(env.state()),grand_grouping(env.state())]
    first=paired_rollouts(snapshot,plans,branch_seeds=[1,2])
    second=paired_rollouts(snapshot,list(reversed(plans)),branch_seeds=[2,1])
    assert first[0]['outcomes']==list(reversed(second[1]['outcomes']))
    assert before==pickle.dumps(env.snapshot())
    assert ambient==pickle.dumps(np.random.get_state())
    assert all(isinstance(b,dict) for b in first[0]['branches'])


def test_episode_transaction_recovers_after_index_append_failure(tmp_path,monkeypatch):
    import open_score.research_v5.evaluate as evaluation
    ctx=context(tmp_path)
    original=evaluation.append
    def fail_index(path,row):
        if str(path).endswith('episodes.jsonl'):
            raise OSError('injected index failure after atomic family commit')
        original(path,row)
    monkeypatch.setattr(evaluation,'append',fail_index)
    with pytest.raises(OSError):
        ctx.evaluate(RulePolicy(),'rule')
    monkeypatch.setattr(evaluation,'append',original)
    summary=ctx.evaluate(RulePolicy(),'rule')
    assert summary['complete'] and summary['episodes']==1
    shards=list((ctx.output/'evaluations/rule/test/latest/families').glob('*.gz'))
    assert len(shards)==1
    before=shards[0].read_bytes()
    ctx.evaluate(RulePolicy(),'rule')
    assert shards[0].read_bytes()==before
    with gzip.open(shards[0],'rt',encoding='utf-8') as stream:
        rows=[json.loads(x) for x in stream]
    events=[r['event_id'] for r in rows if r['record_type']=='event']
    assert len(events)==len(set(events))


def test_all_zero_sparse_comparison_has_nonzero_uncertainty():
    rows=[dict(family_id=str(i),success_native=False) for i in range(100)]
    result=paired_interval(rows,rows)
    assert result['difference']==0
    assert result['interval95'][0]<0<result['interval95'][1]


def test_frozen_source_cannot_be_relabelled_by_prepare(tmp_path,monkeypatch):
    import open_score.research_v5.orchestrate as orchestration
    config=defaults(smoke=True)
    config.update(bc_episodes=0,diagnostic_states=0)
    identity=dict(source_hash='first',git_commit='test',source_files={})
    monkeypatch.setattr(orchestration,'source_identity',lambda:identity)
    monkeypatch.setattr(orchestration,'hardware_profile',lambda:{'test':True})
    orchestration.prepare(tmp_path,config,freeze=True)
    manifest=(tmp_path/'shared/budget_manifest.json').read_bytes()
    identity['source_hash']='changed'
    with pytest.raises(ValueError,match='Frozen source'):
        orchestration.prepare(tmp_path,config,freeze=True)
    assert (tmp_path/'shared/budget_manifest.json').read_bytes()==manifest
    with pytest.raises(ValueError,match='source/config'):
        orchestration.execute_task(tmp_path,'T1')


def test_task_direct_entry_requires_frozen_budget(tmp_path):
    from open_score.research_v5.orchestrate import execute_task
    context(tmp_path)
    with pytest.raises(ValueError,match='Freeze'):
        execute_task(tmp_path,'T1')


def test_gpu_lock_release_retries_windows_reader_sharing_violation(tmp_path,monkeypatch):
    from pathlib import Path
    ctx=context(tmp_path);ctx.device='cuda'
    original=Path.unlink;attempts=[]
    def transient_reader(path,*args,**kwargs):
        if path.name=='.gpu.lock':
            attempts.append(path)
            if len(attempts)<3:
                raise PermissionError('Windows reader temporarily denies delete sharing')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',transient_reader)
    with ctx.gpu():
        assert (tmp_path/'shared/.gpu.lock').exists()
    assert len(attempts)==3
    assert not (tmp_path/'shared/.gpu.lock').exists()


def _lock_contender(directory,active,overlap):
    ctx=TaskContext(directory,'lock_'+str(os.getpid()))
    ctx.device='cuda'
    for _ in range(30):
        with ctx.gpu():
            with active.get_lock():
                active.value+=1
                if active.value!=1:
                    overlap.value+=1
            time.sleep(.002)
            with active.get_lock():
                active.value-=1


def test_gpu_admission_with_real_spawned_contenders(tmp_path):
    context(tmp_path)
    spawn=mp.get_context('spawn');active=spawn.Value('i',0);overlap=spawn.Value('i',0)
    processes=[spawn.Process(target=_lock_contender,args=(str(tmp_path),active,overlap)) for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(40)
        assert process.exitcode==0
    assert overlap.value==0 and active.value==0
    assert not (tmp_path/'shared/.gpu.lock').exists()
