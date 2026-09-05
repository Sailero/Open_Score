from audit_execution import *

def install(env,mode):
    original=env.adapter.local_observation
    def local(side,target_id,red_ids,blue_ids,local_step=0):
        selected=blue_ids
        if mode=='nearest3_full_count':
            r,b=env.adapter.agent_states('Red'),env.adapter.agent_states('Blue')
            center=np.mean([r[i]['position'] for i in red_ids],axis=0)
            selected=sorted(blue_ids,key=lambda i:np.linalg.norm(b[i]['position']-center))[:3]
        row=original(side,target_id,red_ids,selected,local_step=local_step)
        row['self_obs'][:,8]=np.log1p(min(3,len(blue_ids)) if mode=='full_clamped_count' else len(blue_ids))
        return row
    env.adapter.local_observation=local

rows=[];begin=time.monotonic();out=Path('outputs/overnight_audit/count_ablation.json')
for scale in [8,12,16]:
 for mode in ['full_clamped_count','nearest3_full_count']:
    env=KnownOpponentEnv(scale,scale);install(env,mode);subset=[]
    for index in range(70):
        seed=76100030+index;state=env.reset(seed)
        while not env.done:state,reward,done,info=env.step(compact(state))
        row={'scale':scale,'mode':mode,'seed':seed,'success':info['success'],'steps':state.step,'remaining_red':info['remaining_red'],'remaining_blue':info['remaining_blue']};rows.append(row);subset.append(row)
        out.write_text(json.dumps({'protocol':'counterfactual_lower_input_only_same_physics','elapsed':time.monotonic()-begin,'rows':rows},indent=2),encoding='utf8')
    print(json.dumps({'scale':scale,'mode':mode,'wins':sum(x['success'] for x in subset),'episodes':len(subset)}),flush=True)
