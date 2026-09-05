from audit_execution import *
from collections import Counter

rows=[]
for scale in [8,12,16]:
 for mode in ['all','nearest3','support']:
    env=KnownOpponentEnv(scale,scale);patch_observation(env,mode)
    for index in range(30):
        state=env.reset(76100000+index);action=compact(state);assign=action.assignment();physical=env.adapter
        selected=env.executor.act(physical,action)
        rp={x.id:np.array(x.position) for x in state.red};bp=np.array([x.position for x in state.blue]);tp={x.id:np.array(x.position) for x in state.targets}
        for i in rp:
            v=physical.action_vectors[selected[i]];enemy=bp[np.argmin(np.linalg.norm(bp-rp[i],axis=1))]-rp[i];target=tp[assign[i]]-rp[i]
            rows.append({'scale':scale,'mode':mode,'seed':76100000+index,'red_id':i,'action':selected[i],'direction':v.tolist(),'cos_enemy':float(np.dot(v,enemy)/max(np.linalg.norm(enemy),1e-9)),'cos_target':float(np.dot(v,target)/max(np.linalg.norm(target),1e-9))})
out={'rows':rows,'summary':[]}
for scale in [8,12,16]:
 for mode in ['all','nearest3','support']:
    subset=[r for r in rows if r['scale']==scale and r['mode']==mode]
    out['summary'].append({'scale':scale,'mode':mode,'mean_cos_enemy':np.mean([r['cos_enemy'] for r in subset]),'mean_cos_target':np.mean([r['cos_target'] for r in subset]),'mean_acceleration':np.mean([r['direction'] for r in subset],axis=0).tolist(),'action_histogram':dict(Counter(r['action'] for r in subset))})
Path('outputs/overnight_audit/initial_actions.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out['summary'],indent=2))
