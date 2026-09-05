"""Ceiling diagnostic only: replaces frozen acceleration execution, never physics."""
from audit_execution import *

def install(env,mode):
    def act(adapter, grouping):
        reds={i:r for i,r in adapter.agent_states('Red').items() if r['alive']}
        blues={i:b for i,b in adapter.agent_states('Blue').items() if b['alive']}
        results={i:0 for i in adapter.red_ids}
        if not reds or not blues:return results
        ri,bi=list(reds),list(blues)
        center=np.array([blues[i]['position']+2*blues[i]['velocity'] for i in bi])
        rp=np.array([reds[i]['position'] for i in ri])
        if mode=='nearest':
            assigned={i:int(np.argmin(np.linalg.norm(center-rp[j],axis=1))) for j,i in enumerate(ri)}
        else:
            costs=np.linalg.norm(rp[:,None]-center[None],axis=-1)
            rows,cols=linear_sum_assignment(costs)
            assigned={ri[r]:int(c) for r,c in zip(rows,cols)}
            for i in ri:
                if i not in assigned:assigned[i]=int(np.argmin(np.linalg.norm(center-reds[i]['position'],axis=1)))
        for i,j in assigned.items():
            dest=center[j]
            direction=dest-reds[i]['position']
            if mode!='pursuit':direction=250*direction/max(np.linalg.norm(direction),1e-9)-reds[i]['velocity']
            results[i]=env.executor._nearest_action(adapter,direction)
        env.executor.last_actions.update(results)
        return results
    env.executor.act=act

def run_rule():
    out=Path('outputs/overnight_audit/rule_executor.json');rows=[];started=time.monotonic()
    for scale in [8,12,16]:
      for mode in ['nearest','assignment','pursuit']:
        env=KnownOpponentEnv(scale,scale,opponent='reactive');install(env,mode)
        subset=[]
        for index in range(50):
            state=env.reset(76100000+index)
            while not env.done:state,reward,done,info=env.step(compact(state))
            row={'scale':scale,'executor':mode,'seed':76100000+index,'success':info['success'],'steps':state.step,'remaining_red':info['remaining_red'],'remaining_blue':info['remaining_blue']};rows.append(row);subset.append(row)
            out.write_text(json.dumps({'protocol':'diagnostic_changed_lower_executor_same_physics','elapsed':time.monotonic()-started,'rows':rows},indent=2),encoding='utf8')
        print(json.dumps({'scale':scale,'executor':mode,'wins':sum(x['success'] for x in subset),'episodes':len(subset),'mean_steps':np.mean([x['steps'] for x in subset])}),flush=True)

if __name__=='__main__':run_rule()
