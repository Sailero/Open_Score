"""Short real-environment runs and exact completed-job recovery."""
from contextlib import nullcontext
import pickle

import pytest
import torch

from open_score.grouping.storage import atomic_checkpoint
from open_score.research_v4.actions import rule_grouping
from open_score.research_v5.protocol import defaults,episode_spec
from open_score.research_v5.simulator import make_env
from open_score.research_v5.neural import load_policy
from open_score.research_v5.tasks.t3_ppo import run as ppo_run,train_arm
from open_score.research_v5.tasks.t4_exit import run as exit_run,teacher_targets,ExitTeacher


class TinyContext:
    def __init__(self,path,task):
        self.output=path
        path.mkdir(parents=True,exist_ok=True)
        self.task_id=task
        self.seed=401
        self.device='cpu'
        self.config=defaults(True,seed=self.seed)
        self.config['cells']=[[4,2]]
        self.config['t3'].update(steps=35,rollout=8,batch_size=4,epochs=1,candidates=4)
        self.config['t4'].update(rounds=1,states_per_round=2,candidates=4,branches=1,epochs=1,batch_size=2)
        self.config['own_diagnostic_states']=0
        self.messages=[]

    def gpu(self):
        return nullcontext()

    def progress(self,*args,**kwargs):
        self.messages.append(('progress',args,kwargs))

    def log(self,*args):
        self.messages.append(('log',args))

    def checkpoint(self,name,payload):
        atomic_checkpoint(self.output/name,payload)
        return self.output/name

    def load_checkpoint(self,name):
        path=self.output/name
        return torch.load(path,map_location='cpu',weights_only=False) if path.exists() else None

    def spec(self,index,split='train',namespace=None):
        return episode_spec(self.seed,index,split,namespace or self.task_id,self.config['cells'])

    def make_env(self,index,split='train',namespace=None):
        return make_env(self.spec(index,split,namespace))

    def bc_records(self):
        rows=[]
        for index in range(2):
            env=self.make_env(index,namespace='BC')
            state=env.state()
            rows.append(dict(state=state,action=rule_grouping(state),family_id=env.episode_spec.family_id))
        return rows

    def evaluate(self,policy,method_id,**kwargs):
        # Concrete legal action and pickling, while keeping tests independent of full test quotas.
        env=self.make_env(100,'validation')
        restored=pickle.loads(pickle.dumps(policy))
        action=restored.act_env(env) if hasattr(restored,'act_env') else restored.act(env.state())
        action.validate(env.state().ids('red'),env.state().ids('targets'),max_members=None)
        return dict(wins=0,episodes=1,success_rate=0.)

    def diagnose(self,*args,**kwargs):
        return dict(complete=True)

    def collect_states(self,count,split,policy=None,namespace=None):
        result=[]
        for index in range(count):
            env=self.make_env(index,split,namespace)
            result.append(dict(state=env.state(),snapshot=env.snapshot(),state_id=f'{namespace}_{index}',
                family_id=env.episode_spec.family_id,episode_spec=env.episode_spec.to_dict(),collection_physical_steps=0))
        return result


def test_teacher_all_ties_preserve_baseline_and_non_ties_keep_all_winners():
    assert teacher_targets([0,0,0],2)==([0.,0.,1.],True)
    assert teacher_targets([0,.5,.5],0)==([0.,.5,.5],False)


def test_two_ppo_arms_complete_native_episodes_and_resume_without_updates(tmp_path):
    ctx=TinyContext(tmp_path/'T3','T3')
    result=ppo_run(ctx)
    assert result['complete'] and len(result['arms'])==2
    for arm in result['arms']:
        assert 35<=arm['physical_steps']<85
        assert arm['bc_updates']>0 and arm['critic_updates']>0
        checkpoint=ctx.load_checkpoint(arm['method_id'].removeprefix('t3_')+'/latest.pt')
        assert not checkpoint['pending'] and checkpoint['environment'] is None
        assert checkpoint['entropy_objective']==('candidate_distribution_entropy' if arm['method_id']=='t3_candidate' else 'mean_conditional_entropy_per_live_member')
    event_rows=[m[1][1] for m in ctx.messages if m[0]=='log' and m[1][0]=='events']
    assert event_rows
    for row in event_rows:
        assert row['per_member_entropy']==pytest.approx(row['joint_entropy']/max(1,row['live_red']))
        if row['method_id']=='t3_candidate':
            assert row['entropy']==row['joint_entropy'] and row['conditional_entropy'] is None
        else:
            assert row['entropy']==pytest.approx(row['per_member_entropy'])
    before=[ctx.load_checkpoint(f'{k}/latest.pt')['actor'] for k in ('candidate','autoregressive')]
    again=ppo_run(ctx)
    assert [r['actor_updates'] for r in result['arms']]==[r['actor_updates'] for r in again['arms']]
    for old,kind in zip(before,('candidate','autoregressive')):
        latest=ctx.load_checkpoint(f'{kind}/latest.pt')['actor']
        assert all(torch.equal(value,latest[key]) for key,value in old.items())


def test_exit_independent_teacher_and_round_checkpoint_recovery(tmp_path):
    ctx=TinyContext(tmp_path/'T4','T4')
    result=exit_run(ctx)
    assert result['complete'] and result['rounds_completed']==1
    assert result['teacher_states']==2 and result['teacher_physical_steps']>0
    assert result['distillation_updates']>0
    policy=load_policy(ctx.output/'latest.pt')
    assert policy.kind=='candidate'
    again=exit_run(ctx)
    assert result==again


def test_ppo_pending_rollout_and_physical_rng_survive_interruption(tmp_path,monkeypatch):
    from open_score.research_v5.tasks import t3_ppo
    clock=[0.]
    def tick():
        clock[0]+=100.
        return clock[0]
    monkeypatch.setattr(t3_ppo.time,'monotonic',tick)
    golden=TinyContext(tmp_path/'golden','T3')
    train_arm(golden,'autoregressive')
    interrupted=TinyContext(tmp_path/'interrupted','T3')
    original=interrupted.checkpoint
    def crash_after_commit(name,payload):
        result=original(name,payload)
        if name=='autoregressive/latest.pt' and payload['phase']=='ppo' and payload['pending']:
            raise RuntimeError('simulated process interruption')
        return result
    interrupted.checkpoint=crash_after_commit
    with pytest.raises(RuntimeError,match='simulated process interruption'):
        train_arm(interrupted,'autoregressive')
    pending=interrupted.load_checkpoint('autoregressive/latest.pt')
    assert pending['pending']
    interrupted.checkpoint=original
    train_arm(interrupted,'autoregressive')
    expected=golden.load_checkpoint('autoregressive/latest.pt')
    actual=interrupted.load_checkpoint('autoregressive/latest.pt')
    assert expected['counts']==actual['counts']
    for component in ('actor','critic'):
        assert all(torch.equal(value,actual[component][key]) for key,value in expected[component].items())
