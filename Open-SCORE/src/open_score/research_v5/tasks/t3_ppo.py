"""Matched candidate and full constructive PPO with resumable event rollouts."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Categorical

from open_score.grouping.storage import random_state, restore_random_state, seed_everything
from ..neural import (make_actor, StateCritic, NeuralPolicy, CoverageProvider, load_policy,
                      imitation_update, ppo_update)
from ..planning import propose_plans
from ..protocol import stable_seed
from ..simulator import env_from_snapshot


def optimizer_to(optimizer,device):
    for values in optimizer.state.values():
        for key,value in values.items():
            if torch.is_tensor(value):
                values[key] = value.to(device)


@contextmanager
def update_device(ctx,models,optimizers):
    """CPU sampling; lock only a short GPU optimization block."""
    with ctx.gpu():
        try:
            for model in models:
                model.to(ctx.device)
            for optimizer in optimizers:
                optimizer_to(optimizer,ctx.device)
            yield
        finally:
            for model in models:
                model.to('cpu')
            for optimizer in optimizers:
                optimizer_to(optimizer,'cpu')


def policy_payload(actor,kind,budget,**values):
    return dict(schema='v5-neural-checkpoint',kind=kind,model_config=actor.config,
        actor=copy.deepcopy(actor.state_dict()),candidate_budget=int(budget),proposal_seed=0,**values)


def bc_rows(ctx,budget):
    result = []
    for row in ctx.bc_records():
        state = row['state']
        teacher = row.get('action',row.get('teacher'))
        result.append(dict(state=state,teacher=teacher,candidates=propose_plans(state,budget,0),
                           family_id=row['family_id']))
    return result


def validation_rate(summary):
    if 'success_rate' in summary:
        return float(summary['success_rate'])
    if 'wins' in summary and 'episodes' in summary:
        return summary['wins']/max(1,summary['episodes'])
    for key in ('groups','cells','by_cell','results'):
        groups = summary.get(key)
        if isinstance(groups,list) and groups:
            return sum(r.get('wins',r.get('successes',0)) for r in groups)/max(1,sum(r.get('episodes',0) for r in groups))
    if isinstance(summary.get('overall'),dict):
        return validation_rate(summary['overall'])
    raise ValueError(f'Cannot identify validation success metric from {list(summary)}')


def train_arm(ctx,kind):
    config = dict(ctx.config['t3'])
    config['rollout_events'] = int(config.get('rollout',1024))
    budget = int(config['candidates'])
    ctx.select_method(f't3_{kind}')
    directory = ctx.output/'models'
    directory.mkdir(parents=True,exist_ok=True)
    restored = ctx.load_checkpoint('resume.pt')
    if restored is None and (directory/'final.pt').exists():
        restored = ctx.load_checkpoint('final.pt')
    seed_everything(stable_seed(ctx.seed,'T3','matched_initialization'))
    actor = make_actor(kind,ctx.config['model'])
    torch.manual_seed(stable_seed(ctx.seed,'T3','matched_critic_initialization'))
    critic = StateCritic(**ctx.config['model'])
    actor_optimizer = torch.optim.Adam(actor.parameters(),lr=float(config['learning_rate']))
    critic_optimizer = torch.optim.Adam(critic.parameters(),lr=float(config['learning_rate']))
    counts = dict(physical_steps=0,episodes=0,wins=0,upper_events=0,actor_updates=0,
                  critic_updates=0,bc_updates=0,bc_epoch=0,rollouts=0)
    phase,next_episode,pending,env,validated = 'bc',0,[],None,[]
    best_rate = -1.
    recent,started = [],time.monotonic()
    elapsed = 0.
    if restored:
        if restored.get('training_seed', ctx.seed) != ctx.seed or restored['kind'] != kind:
            raise ValueError('The saved PPO seed or policy kind differs from this run')
        for key in ('steps', 'rollout', 'batch_size', 'epochs', 'learning_rate', 'candidates'):
            if key in restored.get('config', {}) and restored['config'][key] != config[key]:
                raise ValueError(f'The saved PPO parameter {key} differs from this run')
        actor.load_state_dict(restored['actor'])
        critic.load_state_dict(restored['critic'])
        actor_optimizer.load_state_dict(restored['actor_optimizer'])
        critic_optimizer.load_state_dict(restored['critic_optimizer'])
        counts.update(restored['counts'])
        phase,next_episode,pending = restored['phase'],restored['next_episode'],restored['pending']
        validated,best_rate = restored['validated'],restored['best_validation_rate']
        recent,elapsed = restored.get('recent',[]),restored.get('training_seconds',0.)
        if restored.get('environment') is not None:
            env = env_from_snapshot(restored['environment'])
            env.episode_spec = ctx.spec(restored['episode_index'],'train','T3_matched')
        restore_random_state(restored['rng'])

    def save(name='latest.pt'):
        payload = policy_payload(actor,kind,budget,critic=copy.deepcopy(critic.state_dict()),
            actor_optimizer=copy.deepcopy(actor_optimizer.state_dict()),critic_optimizer=copy.deepcopy(critic_optimizer.state_dict()),
            counts=copy.deepcopy(counts),phase=phase,next_episode=next_episode,pending=pending,
            environment=env.snapshot() if env is not None else None,episode_index=next_episode-1,
            rng=random_state(),validated=list(validated),best_validation_rate=best_rate,recent=list(recent),
            training_seconds=elapsed+time.monotonic()-started,
            entropy_objective=('mean_conditional_entropy_per_live_member' if kind=='autoregressive' else 'candidate_distribution_entropy'),config=config,
            initialization='shared_rule_BC',training_seed=ctx.seed,
            protocol_version=ctx.config.get('version'),protocol_hash=getattr(ctx,'identity',{}).get('protocol_hash'),
            opponent=ctx.config.get('opponent'),executor=ctx.config.get('executor'))
        return ctx.checkpoint(name,payload)

    if phase=='bc':
        prepared=time.monotonic()
        teacher_rows = bc_rows(ctx,budget)
        ctx.log('phase_costs',dict(phase='bc_prepare',method_id=f't3_{kind}',wall_s=time.monotonic()-prepared,
            states=len(teacher_rows),optimizer_steps=0,real_physical_steps=0,simulation_physical_steps=0))
        for epoch in range(int(counts['bc_epoch']),int(ctx.config['bc_epochs'])):
            epoch_started=time.monotonic()
            updates_before=counts['bc_updates']
            permutation = np.random.permutation(len(teacher_rows)).tolist()
            losses = []
            for start in range(0,len(permutation),config['batch_size']):
                rows = [teacher_rows[i] for i in permutation[start:start+config['batch_size']]]
                with update_device(ctx,[actor],[actor_optimizer]):
                    metrics = imitation_update(actor,actor_optimizer,rows,kind,config['max_gradient_norm'])
                counts['bc_updates'] += metrics['optimizer_steps']
                losses.append(metrics['loss'])
            counts['bc_epoch']=epoch+1
            ctx.log('training',dict(method_id=f't3_{kind}',phase='rule_bc',epoch=epoch+1,
                loss=float(np.mean(losses)),optimizer_updates=counts['bc_updates'],teacher_examples=len(teacher_rows)))
            ctx.log('phase_costs',dict(phase='rule_bc',method_id=f't3_{kind}',wall_s=time.monotonic()-epoch_started,
                states=len(teacher_rows),optimizer_steps=counts['bc_updates']-updates_before,
                real_physical_steps=0,simulation_physical_steps=0))
            ctx.progress('rule_bc',method=kind,epoch=epoch+1,epochs=ctx.config['bc_epochs'])
            save()
        phase='ppo'
        save()

    policy = NeuralPolicy(actor,kind,budget)
    fractions = config.get('validation_fractions',[0.,.5,1.])

    def validate_due():
        nonlocal best_rate
        due = [f for f in fractions if f not in validated and counts['physical_steps']>=int(f*config['steps'])]
        if not due:
            return
        # Several thresholds crossed by a final episode share one real checkpoint.
        checkpoint = f'step_{counts["physical_steps"]}'
        save()
        result = ctx.evaluate(policy,f't3_{kind}',split='validation',checkpoint=checkpoint)
        rate = validation_rate(result)
        if rate>best_rate:
            best_rate=rate
            save('best.pt')
        validated.extend(due)
        ctx.log('validation',dict(method_id=f't3_{kind}',physical_steps=counts['physical_steps'],success_rate=rate,checkpoint=checkpoint))
        save()

    if phase=='ppo':
        validate_due()
        last_save = time.monotonic()
        while counts['physical_steps']<config['steps'] or env is not None:
            sample_started=time.monotonic()
            if env is None:
                env = ctx.make_env(next_episode,'train','T3_matched')
                next_episode += 1
            state = env.state()
            with torch.no_grad():
                value = float(critic([state])[0])
                if kind=='autoregressive':
                    decision = actor([state])
                    action = decision['plans'][0]
                    trace = decision['traces'][0]
                    logp = float(decision['log_prob'][0])
                    entropy = float(decision['entropy'][0])
                    joint_entropy = float(decision['joint_entropy'][0])
                    extra = dict(trace=trace)
                else:
                    pool = propose_plans(state,budget,0)
                    logits = actor([state],[pool])[0]
                    distribution = Categorical(logits=logits)
                    selected = int(distribution.sample())
                    action = pool[selected]
                    logp = float(distribution.log_prob(torch.tensor(selected)))
                    joint_entropy = float(distribution.entropy())
                    entropy = joint_entropy
                    extra = dict(candidates=pool,action=selected)
                per_member_entropy=joint_entropy/max(1,len(state.ids('red')))
            following,reward,done,info = env.step(action)
            pending.append(dict(state=state,next_state=following,reward=reward,done=done,value=value,
                                log_prob=logp,env=0,physical_delta=int(info['delta']),
                                sampling_wall_s=time.monotonic()-sample_started,**extra))
            counts['physical_steps'] += int(info['delta'])
            counts['upper_events'] += 1
            spec = env.episode_spec
            ctx.log('events',dict(method_id=f't3_{kind}',phase='train',family_id=spec.family_id,
                red_count=spec.red_count,blue_count=spec.blue_count,physical_step=state.step,
                physical_delta=info['delta'],event_type=info['event_reason'],terminal=done,
                native_reward=reward,plan=action,live_red=len(state.ids('red')),reserve_count=len(action.reserve),
                group_count=len(action.groups),joint_log_probability=logp,
                entropy=entropy,conditional_entropy=entropy if kind=='autoregressive' else None,
                joint_entropy=joint_entropy,per_member_entropy=per_member_entropy,legal=True))
            if done:
                counts['episodes']+=1
                counts['wins']+=int(info['success'])
                recent.append(int(info['success']))
                recent=recent[-100:]
                ctx.log('episodes',dict(method_id=f't3_{kind}',phase='train',family_id=spec.family_id,
                    red_count=spec.red_count,blue_count=spec.blue_count,episode=next_episode-1,
                    success_native=bool(info['success']),success=bool(info['success']),physical_steps=following.step,
                    total_training_physical_steps=counts['physical_steps']))
                env.close()
                env=None
            if len(pending)>=config['rollout_events'] or (env is None and counts['physical_steps']>=config['steps']):
                ctx.log('phase_costs',dict(phase='ppo_sample',method_id=f't3_{kind}',
                    wall_s=sum(r.get('sampling_wall_s',0.) for r in pending),states=len(pending),
                    real_physical_steps=sum(r.get('physical_delta',0) for r in pending),
                    simulation_physical_steps=0,optimizer_steps=0,episodes=sum(r['done'] for r in pending)))
                update_started=time.monotonic()
                with update_device(ctx,[actor,critic],[actor_optimizer,critic_optimizer]):
                    metrics = ppo_update(actor,critic,actor_optimizer,critic_optimizer,pending,kind,config,
                                         min(1.,counts['physical_steps']/config['steps']))
                ctx.log('phase_costs',dict(phase='ppo_update',method_id=f't3_{kind}',wall_s=time.monotonic()-update_started,
                    states=len(pending),real_physical_steps=0,simulation_physical_steps=0,
                    optimizer_steps=metrics['actor_updates']+metrics['critic_updates']))
                counts['actor_updates']+=metrics['actor_updates']
                counts['critic_updates']+=metrics['critic_updates']
                counts['rollouts']+=1
                pending=[]
                ctx.log('training',dict(method_id=f't3_{kind}',phase='ppo',physical_steps=counts['physical_steps'],
                    **metrics,success_window=float(np.mean(recent)) if recent else None))
                save()
                # Validation cannot disturb an in-flight sampled policy likelihood.
                validate_due()
            if counts['upper_events']%16==0 or time.monotonic()-last_save>30:
                ctx.progress('ppo',method=kind,physical_steps=counts['physical_steps'],total_physical_steps=config['steps'],
                    episodes=counts['episodes'],recent_success=float(np.mean(recent)) if recent else None,
                    current_member_deployment_rate=1-len(action.reserve)/max(1,len(state.ids('red'))),
                    actor_updates=counts['actor_updates'],critic_updates=counts['critic_updates'])
            if time.monotonic()-last_save>60:
                save()
                last_save=time.monotonic()
        phase='evaluation'
        save()
    if phase=='evaluation':
        validate_due()
        ctx.evaluate(load_policy(directory/'resume.pt'),f't3_{kind}',checkpoint='latest')
        phase='complete'
        save('final.pt')
        (directory/'resume.pt').unlink(missing_ok=True)
    result = dict(method_id=f't3_{kind}',complete=phase=='complete',**counts,
                best_validation_rate=best_rate,model_parameters=sum(p.numel() for p in actor.parameters()),
                critic_parameters=sum(p.numel() for p in critic.parameters()),
                latest=str(directory/'final.pt'),best=str(directory/'best.pt'))
    ctx.store.put('result',result)
    return result


def run(ctx):
    torch.set_num_threads(1)
    results=[]
    kinds = ('candidate','autoregressive') if ctx.config.get('selected_method') not in ('candidate_ppo','ar_ppo') else (
        ('candidate',) if ctx.config['selected_method']=='candidate_ppo' else ('autoregressive',))
    for kind in kinds:
        results.append(train_arm(ctx,kind))
    ctx.progress('complete',complete=True,results=results)
    return dict(complete=True,task='T3',arms=results)


