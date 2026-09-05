"""Read-only core diagnostic: policies/observation alternatives, same physics."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from open_score.grouping.environment import KnownOpponentEnv
from open_score.grouping.domain import Group, Grouping
from open_score.grouping.baselines import StaticPolicy, balanced_initial
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def compact(state, threat=False, size=4):
    red, blue, targets=state.alive('red'),state.alive('blue'),state.alive('targets')
    n=len(red)
    if not n:return Grouping(())
    if threat:
        weights=np.array([sum(np.exp(-np.linalg.norm(np.array(b.position)+2*np.array(b.velocity)-t.position)/850) for b in blue) for t in targets])
        weights=.25/len(targets)+.75*weights/max(weights.sum(),1e-9)
        quotas=np.floor(weights*n).astype(int)
        while quotas.sum()<n:quotas[np.argmax(weights*n-quotas)]+=1
        slots=[t for t,q in zip(targets,quotas) for _ in range(q)]
    else:slots=[targets[i%len(targets)] for i in range(n)]
    cost=np.array([[np.linalg.norm(np.array(r.position)-t.position) for t in slots] for r in red])
    rows,cols=linear_sum_assignment(cost)
    mapping={red[i].id:slots[j].id for i,j in zip(rows,cols)}
    records={r.id:r for r in red}
    groups=[]
    for t in targets:
        remaining=[i for i in records if mapping[i]==t.id]
        while remaining:
            first=min(remaining,key=lambda i:np.linalg.norm(np.array(records[i].position)-t.position))
            cluster=sorted(remaining,key=lambda i:np.linalg.norm(np.array(records[i].position)-records[first].position))[:size]
            groups.append(Group(t.id,tuple(cluster)))
            remaining=[i for i in remaining if i not in cluster]
    return Grouping(tuple(groups))

def patch_observation(env, mode):
    if mode=='all':return
    original=env.adapter.local_observation
    fixed_k=None if mode=='support' else int(mode.removeprefix('nearest'))
    def local(side,target_id,red_ids,blue_ids,local_step=0):
        reds,blues=env.adapter.agent_states('Red'),env.adapter.agent_states('Blue')
        center=np.mean([reds[i]['position'] for i in red_ids],axis=0)
        k=min(3,max(1,len(red_ids)-1)) if fixed_k is None else fixed_k
        selected=sorted(blue_ids,key=lambda i:np.linalg.norm(blues[i]['position']-center))[:k]
        return original(side,target_id,red_ids,selected,local_step=local_step)
    env.adapter.local_observation=local

def run(args):
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=[];start=time.monotonic()
    for scale in args.scales:
      for mode in args.observations:
       env=KnownOpponentEnv(scale,scale,opponent=args.opponent)
       patch_observation(env,mode)
       for policy in args.policies:
        begin=time.monotonic(); subset=[]
        for index in range(args.episodes):
            seed=args.seed+index
            state=env.reset(seed);events=[];decisions=0
            static=StaticPolicy()
            while not env.done:
                if policy=='static':action=static.act(state).action
                elif policy=='reserve':action=Grouping((),state.ids('red'))
                elif policy=='balanced':action=balanced_initial(state)
                elif policy=='compact':action=compact(state)
                elif policy=='threat':action=compact(state,True)
                elif policy=='pairs':action=compact(state,True,2)
                else:raise ValueError(policy)
                state,reward,done,info=env.step(action);events+=info['events'];decisions+=1
            record={'scale':scale,'observation':mode,'policy':policy,'seed':seed,'success':info['success'],'steps':state.step,'decisions':decisions,'red':info['remaining_red'],'blue':info['remaining_blue'],'target_health':[x.health for x in state.targets],'event_kinds':[x['kind'] for x in events]}
            subset.append(record);rows.append(record)
            with output.open('w',encoding='utf8') as f:json.dump({'config':vars(args),'elapsed':time.monotonic()-start,'rows':rows},f,indent=2)
        print(json.dumps({'scale':scale,'mode':mode,'policy':policy,'wins':sum(x['success'] for x in subset),'episodes':len(subset),'mean_steps':np.mean([x['steps'] for x in subset]),'mean_red':np.mean([x['red'] for x in subset]),'mean_blue':np.mean([x['blue'] for x in subset]),'seconds':time.monotonic()-begin}),flush=True)
       env.close()
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=30);p.add_argument('--seed',type=int,default=76100000);p.add_argument('--scales',type=int,nargs='+',default=[8,12,16]);p.add_argument('--observations',nargs='+',default=['all','nearest3']);p.add_argument('--policies',nargs='+',default=['static','reserve','compact','threat','pairs']);p.add_argument('--opponent',default='reactive');p.add_argument('--output',default='outputs/overnight_audit/execution.json');raise SystemExit(run(p.parse_args()))
