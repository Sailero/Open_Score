import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
payload=torch.load(root.parents[1]/'assets/frozen/lcl.pt',map_location='cpu',weights_only=False)
extra=payload['extra']
summary={'lcl':{key:extra[key] for key in ['protocol','architecture','environment_steps','episode_count','train_side']},'source_support':extra['curriculum']['scales'],'checkpoint_selection':extra['selection_evaluation'],'evaluations':{}}
for name in ['execution','compact_confirmation','rule_executor','count_ablation']:
    path=root/f'{name}.json'
    if not path.is_file():continue
    data=json.loads(path.read_text(encoding='utf8'));groups={}
    for row in data['rows']:
        method=row.get('policy',row.get('executor','compact'))
        mode=row.get('observation',row.get('mode','changed_rule_executor'))
        key=f"{row['scale']}v{row['scale']}|{method}|{mode}"
        group=groups.setdefault(key,{'episodes':0,'wins':0,'steps':0,'validation_episodes':0,'validation_wins':0})
        group['episodes']+=1;group['wins']+=bool(row['success']);group['steps']+=row['steps']
        if row['seed']>=76100030:
            group['validation_episodes']+=1;group['validation_wins']+=bool(row['success'])
    for group in groups.values():
        group['success_rate']=group['wins']/group['episodes'];group['mean_steps']=group.pop('steps')/group['episodes']
    summary['evaluations'][name]={'elapsed_seconds':data['elapsed'],'rows':len(data['rows']),'groups':groups}
(root/'failure_audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps({name:record['groups'] for name,record in summary['evaluations'].items()},indent=2))
