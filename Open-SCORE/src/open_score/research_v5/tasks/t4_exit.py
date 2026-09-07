"""Independent ExIt: frozen-policy terminal search followed by distillation."""
from __future__ import annotations

import copy
import io
import hashlib
import time

import numpy as np
import torch

from open_score.grouping.storage import random_state, restore_random_state, seed_everything
from open_score.research_v4.actions import rule_grouping,partition_key
from ..neural import CandidateActor,NeuralPolicy,load_policy,imitation_update
from ..planning import propose_plans,choose_with_rule_ties
from ..protocol import stable_seed
from ..simulator import paired_rollouts
from .t3_ppo import update_device,policy_payload,bc_rows,validation_rate


def frozen_version(actor,round_index):
    return f'exit_round_{round_index}'


class ExitTeacher:
    """Picklable CPU teacher; never owns a context or a real future RNG."""
    def __init__(self,continuation=None,candidates=8,branches=8,seed=0,version='rule_grouping_v1'):
        self.continuation=continuation
        self.candidates,self.branches,self.seed=int(candidates),int(branches),int(seed)
        self.version=version
        self.last_trace={}

    def act_env(self,env):
        state=env.state()
        pool=propose_plans(state,self.candidates,0)
        seed=stable_seed('T4_online_teacher',self.seed,state.to_dict())
        rows=paired_rollouts(env.snapshot(),pool,self.continuation,
            [stable_seed(seed,b) for b in range(self.branches)],self.version)
        selected=choose_with_rule_ties(state,pool,[r['y'] for r in rows])
        self.last_trace=dict(candidate_evaluations=len(pool),simulation_physical_steps=sum(sum(r['physical_steps']) for r in rows),
            branch_count=len(pool)*self.branches,selected_index=selected,
            candidate_success=[r['y'] for r in rows],continuation_version=self.version,candidates=rows)
        return pool[selected]


def teacher_targets(values,baseline_index):
    values=np.asarray(values,float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite terminal estimates are required')
    ambiguous=bool(np.ptp(values)<=1e-12)
    target=np.zeros(len(values),float)
    if ambiguous:
        target[baseline_index]=1.
    else:
        target[np.abs(values-values.max())<=1e-12]=1.
        target/=target.sum()
    return target.tolist(),ambiguous


def run(ctx):
    torch.set_num_threads(1)
    config=dict(ctx.config['t4'])
    config['teacher_bc_fraction']=.5
    seed_everything(stable_seed(ctx.seed,'T4','initialization'))
    actor=CandidateActor(**ctx.config['model'])
    optimizer=torch.optim.Adam(actor.parameters(),lr=float(config['learning_rate']))
    budget=int(config['candidates'])
    phase='bc'
    round_index=epoch=bc_epoch=0
    labels,records=[],[]
    continuation=None
    continuation_state=None
    continuation_version='rule_grouping_v1'
    best_rate=-1.
    counts=dict(bc_updates=0,distillation_updates=0,teacher_physical_steps=0,
        verification_physical_steps=0,collection_physical_steps=0,teacher_states=0,
        improvement_teacher=0,fallback_imitation=0,bc_examples_consumed=0,teacher_examples_consumed=0)
    restored=ctx.load_checkpoint('latest.pt')
    if restored:
        actor.load_state_dict(restored['actor'])
        optimizer.load_state_dict(restored['optimizer'])
        phase=restored['phase']
        round_index,epoch,bc_epoch=restored['round_index'],restored['epoch'],restored['bc_epoch']
        labels,records=restored['labels'],restored['records']
        continuation_state=restored['continuation_actor']
        continuation_version=restored['continuation_version']
        counts.update(restored['counts'])
        best_rate=restored['best_validation_rate']
        restore_random_state(restored['rng'])

    def save(name='latest.pt'):
        return ctx.checkpoint(name,policy_payload(actor,'candidate',budget,optimizer=copy.deepcopy(optimizer.state_dict()),
            phase=phase,round_index=round_index,epoch=epoch,bc_epoch=bc_epoch,labels=labels,records=records,
            continuation_actor=continuation_state,continuation_version=continuation_version,
            counts=copy.deepcopy(counts),rng=random_state(),best_validation_rate=best_rate,config=config,
            training_seed=ctx.seed,protocol_version=ctx.config.get('version'),
            protocol_hash=getattr(ctx,'identity',{}).get('protocol_hash'),
            opponent=ctx.config.get('opponent'),executor=ctx.config.get('executor')))

    teacher_bc=bc_rows(ctx,budget)
    if phase=='bc':
        for ep in range(bc_epoch,int(ctx.config['bc_epochs'])):
            epoch_started=time.monotonic()
            updates_before=counts['bc_updates']
            permutation=np.random.permutation(len(teacher_bc)).tolist()
            losses=[]
            for start in range(0,len(permutation),config['batch_size']):
                batch=[teacher_bc[i] for i in permutation[start:start+config['batch_size']]]
                with update_device(ctx,[actor],[optimizer]):
                    metric=imitation_update(actor,optimizer,batch)
                counts['bc_updates']+=metric['optimizer_steps']
                counts['bc_examples_consumed']+=len(batch)
                losses.append(metric['loss'])
            bc_epoch=ep+1
            ctx.log('training',dict(method_id='t4_exit',phase='rule_bc',epoch=bc_epoch,loss=float(np.mean(losses))))
            ctx.log('phase_costs',dict(phase='rule_bc',method_id='t4_exit',wall_s=time.monotonic()-epoch_started,
                states=len(teacher_bc),optimizer_steps=counts['bc_updates']-updates_before,
                real_physical_steps=0,simulation_physical_steps=0))
            ctx.progress('rule_bc',epoch=bc_epoch,epochs=ctx.config['bc_epochs'])
            save()
        phase='collect'
        save()
    while round_index<int(config['rounds']) and phase!='complete':
        if phase=='collect':
            # pi_0 is the exact rule; later rounds freeze the actual prior student.
            if round_index==0:
                continuation_state=None
                continuation_version='rule_grouping_v1'
                continuation=None
            else:
                continuation_state=copy.deepcopy(actor.state_dict())
                continuation_version=frozen_version(actor,round_index)
                frozen=CandidateActor(**ctx.config['model'])
                frozen.load_state_dict(continuation_state)
                continuation=NeuralPolicy(frozen,'candidate',budget)
            records=ctx.collect_states(config['states_per_round'],'train',continuation,
                namespace=f'T4_round_{round_index}_{continuation_version}')
            counts['collection_physical_steps']+=sum(r.get('collection_physical_steps',0) for r in records)
            labels=[]
            epoch=0
            phase='label'
            save()
        if continuation_state is not None:
            frozen=CandidateActor(**ctx.config['model'])
            frozen.load_state_dict(continuation_state)
            continuation=NeuralPolicy(frozen,'candidate',budget)
        else:
            continuation=None
        if phase=='label':
            for index in range(len(labels),len(records)):
                record=records[index]
                state=record['state']
                pool=propose_plans(state,budget,0)
                baseline=rule_grouping(state) if continuation is None else continuation.act(state)
                baseline_index=next(i for i,p in enumerate(pool) if partition_key(p)==partition_key(baseline))
                seeds=[stable_seed(ctx.seed,'T4','selection',round_index,record['state_id'],b) for b in range(config['branches'])]
                started=time.monotonic()
                results=paired_rollouts(record['snapshot'],pool,continuation,seeds,continuation_version)
                targets,ambiguous=teacher_targets([r['y'] for r in results],baseline_index)
                selected=choose_with_rule_ties(state,pool,[r['y'] for r in results]) if not ambiguous else baseline_index
                row=dict(state=state,candidates=pool,target_distribution=targets,family_id=record['family_id'],
                    state_id=record['state_id'],continuation_version=continuation_version,
                    baseline_index=baseline_index,selected_index=selected,ambiguous=ambiguous,
                    teacher_kind='fallback_imitation' if ambiguous else 'improvement_teacher')
                labels.append(row)
                counts['teacher_states']+=1
                counts['fallback_imitation' if ambiguous else 'improvement_teacher']+=1
                work=sum(sum(r['physical_steps']) for r in results)
                counts['teacher_physical_steps']+=work
                for ci,result in enumerate(results):
                    ctx.log('candidates',dict(method_id='t4_exit',round=round_index+1,state_id=record['state_id'],
                        family_id=record['family_id'],candidate_id=ci,plan=pool[ci],outcomes=result['outcomes'],
                        branch_seeds=seeds,success_mean=result['y'],physical_steps_simulated=sum(result['physical_steps']),
                        continuation_version=continuation_version,selected=ci==selected,baseline=ci==baseline_index,
                        teacher_kind=row['teacher_kind']))
                ctx.log('teacher',dict(method_id='t4_exit',round=round_index+1,state_id=record['state_id'],
                    selected_index=selected,baseline_index=baseline_index,ambiguous=ambiguous,
                    physical_steps_simulated=work,wall_time_s=time.monotonic()-started))
                ctx.log('phase_costs',dict(phase='teacher_label',method_id='t4_exit',round=round_index+1,
                    wall_s=time.monotonic()-started,states=1,real_physical_steps=0,
                    simulation_physical_steps=work,optimizer_steps=0,branches=len(pool)*len(seeds)))
                ctx.progress('teacher_label',round=round_index+1,rounds=config['rounds'],
                    labeled_states=len(labels),total_states=len(records),**counts)
                save()
            phase='distill'
            save()
        if phase=='distill':
            for ep in range(epoch,int(config['epochs'])):
                epoch_started=time.monotonic()
                updates_before=counts['distillation_updates']
                # Fixed equal source mass: BC is initialization support, not the dominant label source.
                order=np.random.permutation(len(labels)).tolist()
                bc_indices=np.random.choice(len(teacher_bc),len(labels),replace=len(teacher_bc)<len(labels)).tolist()
                losses=[]
                half=max(1,int(config['batch_size'])//2)
                for start in range(0,len(labels),half):
                    teacher_batch=[labels[i] for i in order[start:start+half]]
                    bc_batch=[teacher_bc[i] for i in bc_indices[start:start+half]]
                    with update_device(ctx,[actor],[optimizer]):
                        metric=imitation_update(actor,optimizer,teacher_batch+bc_batch)
                    counts['distillation_updates']+=metric['optimizer_steps']
                    counts['teacher_examples_consumed']+=len(teacher_batch)
                    counts['bc_examples_consumed']+=len(bc_batch)
                    losses.append(metric['loss'])
                epoch=ep+1
                ctx.log('training',dict(method_id='t4_exit',phase='distill',round=round_index+1,
                    epoch=epoch,loss=float(np.mean(losses)),teacher_bc_fraction=.5,
                    teacher_examples=len(labels),bc_examples=len(labels),continuation_version=continuation_version))
                ctx.log('phase_costs',dict(phase='distill',method_id='t4_exit',wall_s=time.monotonic()-epoch_started,
                    states=2*len(labels),optimizer_steps=counts['distillation_updates']-updates_before,
                    real_physical_steps=0,simulation_physical_steps=0))
                ctx.progress('distill',round=round_index+1,rounds=config['rounds'],epoch=epoch,epochs=config['epochs'],**counts)
                save()
            phase='validate'
            save()
        if phase=='validate':
            student=NeuralPolicy(actor,'candidate',budget)
            name=f'round_{round_index+1}'
            save()
            result=ctx.evaluate(student,'t4_exit',split='validation',checkpoint=name)
            rate=validation_rate(result)
            if rate>best_rate:
                best_rate=rate
                save('best.pt')
            round_index+=1
            labels,records=[],[]
            phase='collect' if round_index<config['rounds'] else 'evaluation'
            save()
    if phase=='evaluation':
        policy=load_policy(ctx.model_path('resume.pt'))
        ctx.evaluate(policy,'t4_exit',checkpoint='latest')
        phase='complete'
        save('final.pt')
        ctx.model_path('resume.pt').unlink(missing_ok=True)
    ctx.progress('complete',complete=True,rounds_completed=round_index,**counts)
    return dict(task='T4',complete=phase=='complete',rounds_completed=round_index,
        best_validation_rate=best_rate,teacher_bc_fraction=.5,**counts)


