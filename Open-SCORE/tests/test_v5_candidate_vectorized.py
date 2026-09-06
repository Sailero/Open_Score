"""The original scalar candidate computation is an independent equivalence oracle."""
import copy
import math
import time

import numpy as np
import pytest
import torch

from open_score.grouping.domain import Group, Grouping
from open_score.research_v4.environment import make_env
from open_score.research_v5.neural import CandidateActor
from open_score.research_v5.planning import propose_plans


def scalar_forward(self,states,pools):
    if len(states)!=len(pools) or any(not p for p in pools):
        raise ValueError('States require matching nonempty candidate lists')
    enc = self.encoder(states)
    group_inputs, owners, features, metadata = [], [], [], []
    for b,(state,pool) in enumerate(zip(states,pools)):
        for k,plan in enumerate(pool):
            plan.validate(state.ids('red'),state.ids('targets'),max_members=None)
            owner = len(features)
            for rank,group in enumerate(plan.groups):
                ids = [enc.mappings[b][0,i] for i in group.members]
                group_inputs.append(torch.cat((enc.entities[b,ids].mean(0),
                    enc.entities[b,enc.mappings[b][2,group.target]],enc.context.new_tensor([
                        math.log1p(len(ids)),rank/max(1,len(state.ids('red')))]))))
                owners.append(owner)
            reserve = ([enc.mappings[b][0,i] for i in plan.reserve])
            reserve_vector = enc.entities[b,reserve].mean(0) if reserve else enc.context[b]*0
            features.append((enc.context[b],reserve_vector,len(plan.groups),len(plan.reserve)))
            metadata.append((b,k))
    h = enc.context.shape[-1]
    total = len(features)
    means = enc.context.new_zeros(total,h)
    maxima = enc.context.new_full((total,h),-1e9)
    counts = enc.context.new_zeros(total,1)
    if group_inputs:
        g = self.group(torch.stack(group_inputs))
        idx = torch.tensor(owners,device=g.device)
        means = means.index_add(0,idx,g)
        counts = counts.index_add(0,idx,g.new_ones(len(g),1))
        maxima = maxima.scatter_reduce(0,idx[:,None].expand_as(g),g,reduce='amax',include_self=True)
    means = means/counts.clamp_min(1)
    maxima = torch.where(counts>0,maxima,torch.zeros_like(maxima))
    inputs = torch.stack([torch.cat((c,means[i],maxima[i],r,c.new_tensor([math.log1p(n),math.log1p(nr)])))
        for i,(c,r,n,nr) in enumerate(features)])
    values = self.head(inputs).squeeze(-1)
    output = values.new_full((len(states),max(map(len,pools))),-1e9)
    b,k = zip(*metadata)
    return output.index_put((torch.tensor(b,device=values.device),torch.tensor(k,device=values.device)),values)


def mixed_inputs():
    from dataclasses import replace
    rng=np.random.default_rng(730)
    states=[];pools=[]
    for red in (8,16,32):
        state=make_env(red,red//2).reset(87)
        ids=state.ids('red');targets=state.ids('targets')
        plans=[Grouping((),ids),Grouping((Group(targets[0],ids),)),
               Grouping(tuple(Group(targets[i%len(targets)],(identity,)) for i,identity in enumerate(ids)))]
        for i in range(3+red//8):
            shuffled=list(map(int,rng.permutation(ids)))
            reserve=tuple(shuffled[:i%4]);deployed=shuffled[i%4:]
            chunks=np.array_split(deployed,2+i%4)
            plans.append(Grouping(tuple(Group(targets[j%len(targets)],tuple(map(int,chunk)))
                for j,chunk in enumerate(chunks) if len(chunk)),reserve))
        states.append(replace(state,previous=plans[-1]))
        pools.append(plans)
    return states,pools


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('all_reserve',[False,True])
def test_candidate_vectorization_preserves_logits_masks_and_all_parameter_gradients(device,all_reserve):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1);torch.manual_seed(982)
    states,pools=mixed_inputs()
    if all_reserve: pools=[[p[0]] for p in pools]
    vectorized=CandidateActor(128,4,2).to(device)
    reference=copy.deepcopy(vectorized)
    actual=vectorized(states,pools)
    expected=scalar_forward(reference,states,pools)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    weights=torch.arange(actual.numel(),device=device,dtype=actual.dtype).reshape_as(actual)/actual.numel()+.1
    (actual*weights).sum().backward();(expected*weights).sum().backward()
    assert list(vectorized.state_dict())==list(reference.state_dict())
    for (name,a),(other,b) in zip(vectorized.named_parameters(),reference.named_parameters()):
        assert name==other and (a.grad is None)==(b.grad is None),name
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=3e-4,msg=lambda message:f'{name}: {message}')


def benchmark(run_dir,output):
    """Two matched forward/backward/Adam measurements at production batch shape."""
    import json
    from pathlib import Path
    from types import MethodType,SimpleNamespace
    from open_score.research_v5.runtime import TaskContext
    torch.set_num_threads(1);torch.manual_seed(1203)
    states=[];pools=[]
    for i in range(128):
        red=(8,16,32)[i%3]
        state=make_env(red,red//2).reset(1300+i)
        pool=propose_plans(state,32)
        assert len(pool)==32
        states.append(state);pools.append(pool)
    base=CandidateActor(128,4,2)
    lock_context=SimpleNamespace(device='cuda',run_dir=Path(run_dir),_gpu_wait_s=0.)
    phases=[]
    for kind in ('scalar_reference','vectorized'):
        actor=copy.deepcopy(base)
        if kind=='scalar_reference': actor.forward=MethodType(scalar_forward,actor)
        optimizer=torch.optim.Adam(actor.parameters(),lr=3e-4)
        with TaskContext.gpu(lock_context):
            actor.to('cuda')
            for repeat in range(2):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                logits=actor(states,pools)
                loss=-torch.nn.functional.log_softmax(logits,-1)[:,0].mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step()
                torch.cuda.synchronize()
                row=dict(kind=kind,repeat=repeat,wall_s=time.perf_counter()-started,
                    loss=float(loss.detach()),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    batch_states=128,candidates_per_state=32,red_counts=[8,16,32])
                phases.append(row);print(json.dumps(row),flush=True)
            actor.to('cpu')
        del actor,optimizer
        torch.cuda.empty_cache()
    result=dict(phases=phases,gpu_wait_s=lock_context._gpu_wait_s,
        note='Matched production actor update only; two repeats include first-call/Adam initialization. GPU lock acquired; not a replacement for joint six-task calibration.')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    import sys
    benchmark(sys.argv[1],sys.argv[2])
