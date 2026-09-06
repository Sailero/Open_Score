"""V5 variable-roster policy models and event-level learning primitives.

Group ranks expose the deterministic executor's canonical tie-breaking rule.
Constructive decisions do not advance physics; their log probabilities are
summed before PPO clipping. No group capacity or compulsory deployment exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from open_score.grouping.domain import Group, Grouping
from open_score.research_v4.actions import partition_key
from .planning import propose_plans


@dataclass
class Encoded:
    context: torch.Tensor
    entities: torch.Tensor
    mappings: list


class EntityEncoder(nn.Module):
    def __init__(self, hidden_dim=128, heads=4, layers=2):
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim, heads=heads, layers=layers)
        self.hidden_dim = hidden_dim
        self.entity = nn.Sequential(nn.Linear(12, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.context = nn.Linear(5, hidden_dim)
        self.old_group = nn.Sequential(nn.Linear(2*hidden_dim+2, hidden_dim), nn.GELU())
        self.reserve = nn.Parameter(torch.zeros(hidden_dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            hidden_dim, heads, 2*hidden_dim, dropout=0., activation='gelu',
            batch_first=True), layers, enable_nested_tensor=False)

    def forward(self, states):
        if not states:
            raise ValueError('At least one state is required')
        device = next(self.parameters()).device
        width = 1+max(sum(len(s.ids(side)) for side in ('red','blue','targets')) for s in states)
        raw = np.zeros((len(states), width, 12), np.float32)
        padding = np.ones((len(states), width), bool)
        maps, contexts = [], []
        for b, state in enumerate(states):
            mapping, offset = {}, 1
            for kind, side in enumerate(('red', 'blue', 'targets')):
                entities = sorted(state.alive(side), key=lambda e:e.id)
                for rank, entity in enumerate(entities):
                    raw[b,offset,:] = [float(kind==k) for k in range(3)]+[
                        v/2500. for v in entity.position]+[v/500. for v in entity.velocity]+[
                        entity.health/(1.2 if kind==2 else 1.),
                        max(0.,1.-state.step/state.max_steps),rank/max(1,len(entities))]
                    mapping[kind,entity.id] = offset
                    offset += 1
            padding[b,:offset] = False
            maps.append(mapping)
            contexts.append([state.step/state.max_steps,
                *[math.log1p(len(state.ids(k))) for k in ('red','blue','targets')],
                (5-state.step%5)/5.])
        base = self.entity(torch.as_tensor(raw,device=device))
        relation = torch.zeros_like(base)
        group_vectors, member_pairs = [], []
        for b,state in enumerate(states):
            groups = state.previous.prune(state.ids('red')).groups
            for rank,group in enumerate(groups):
                if (2,group.target) not in maps[b]:
                    continue
                members = [maps[b][0,i] for i in group.members]
                group_vectors.append(torch.cat((base[b,members].mean(0),base[b,maps[b][2,group.target]],
                    base.new_tensor([math.log1p(len(members)),rank/max(1,len(state.ids('red')))]))))
                member_pairs.append((b,members))
            for identity in state.previous.reserve:
                if (0,identity) in maps[b]:
                    relation[b,maps[b][0,identity]] = self.reserve
        if group_vectors:
            vectors = self.old_group(torch.stack(group_vectors))
            for vector,(b,members) in zip(vectors,member_pairs):
                relation[b,members] = vector
        tokens = base+relation
        tokens = torch.cat((self.context(base.new_tensor(contexts))[:,None],tokens[:,1:]),1)
        encoded = self.transformer(tokens,src_key_padding_mask=torch.as_tensor(padding,device=device))
        return Encoded(encoded[:,0],encoded,maps)


class CandidateActor(nn.Module):
    def __init__(self, hidden_dim=128, heads=4, layers=2):
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim,heads=heads,layers=layers)
        self.encoder = EntityEncoder(**self.config)
        self.group = nn.Sequential(nn.Linear(2*hidden_dim+2,hidden_dim),nn.GELU())
        self.head = nn.Sequential(nn.Linear(4*hidden_dim+2,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))

    def forward(self,states,pools):
        if len(states)!=len(pools) or any(not p for p in pools):
            raise ValueError('States require matching nonempty candidate lists')
        enc = self.encoder(states)
        # Construct topology on CPU, then aggregate every group/reserve in bulk.
        # Flat indices address the same encoded entity rows as the scalar path.
        members, member_groups, targets, owners, group_sizes, group_features = [],[],[],[],[],[]
        reserves, reserve_owners, reserve_sizes, candidate_states, candidate_slots, features = [],[],[],[],[],[]
        width=enc.entities.shape[1]
        for b,(state,pool) in enumerate(zip(states,pools)):
            red_ids,target_ids=state.ids('red'),state.ids('targets')
            mapping=enc.mappings[b]
            for k,plan in enumerate(pool):
                plan.validate(red_ids,target_ids,max_members=None)
                owner = len(features)
                for rank,group in enumerate(plan.groups):
                    ids=[b*width+mapping[0,i] for i in group.members]
                    member_groups.extend([len(owners)]*len(ids))
                    members.extend(ids)
                    targets.append(b*width+mapping[2,group.target])
                    owners.append(owner)
                    group_sizes.append(len(ids))
                    group_features.append([math.log1p(len(ids)),rank/max(1,len(red_ids))])
                reserves.extend(b*width+mapping[0,i] for i in plan.reserve)
                reserve_owners.extend([owner]*len(plan.reserve))
                reserve_sizes.append(len(plan.reserve))
                features.append([math.log1p(len(plan.groups)),math.log1p(len(plan.reserve))])
                candidate_states.append(b)
                candidate_slots.append(k)
        h = enc.context.shape[-1]
        total = len(features)
        device=enc.context.device
        def indices(values):
            return torch.tensor(values,dtype=torch.long,device=device)
        entities=enc.entities.reshape(-1,h)
        context=enc.context.index_select(0,indices(candidate_states))
        reserve_vectors=context*0
        if reserves:
            reserve_vectors=reserve_vectors.index_add(0,indices(reserve_owners),entities.index_select(0,indices(reserves)))
            reserve_vectors=reserve_vectors/enc.context.new_tensor(reserve_sizes)[:,None].clamp_min(1)
        means = enc.context.new_zeros(total,h)
        maxima = enc.context.new_full((total,h),-1e9)
        counts = enc.context.new_zeros(total,1)
        if owners:
            member_sums=enc.context.new_zeros(len(owners),h).index_add(
                0,indices(member_groups),entities.index_select(0,indices(members)))
            member_means=member_sums/enc.context.new_tensor(group_sizes)[:,None]
            group_inputs=torch.cat((member_means,entities.index_select(0,indices(targets)),
                enc.context.new_tensor(group_features)),dim=1)
            g = self.group(group_inputs)
            idx = indices(owners)
            means = means.index_add(0,idx,g)
            counts = counts.index_add(0,idx,g.new_ones(len(g),1))
            maxima = maxima.scatter_reduce(0,idx[:,None].expand_as(g),g,reduce='amax',include_self=True)
        means = means/counts.clamp_min(1)
        maxima = torch.where(counts>0,maxima,torch.zeros_like(maxima))
        inputs=torch.cat((context,means,maxima,reserve_vectors,enc.context.new_tensor(features)),dim=1)
        values = self.head(inputs).squeeze(-1)
        output = values.new_full((len(states),max(map(len,pools))),-1e9)
        return output.index_put((indices(candidate_states),indices(candidate_slots)),values)


class StateCritic(nn.Module):
    def __init__(self,hidden_dim=128,heads=4,layers=2):
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim,heads=heads,layers=layers)
        self.encoder = EntityEncoder(**self.config)
        self.value = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))

    def forward(self,states):
        return self.value(self.encoder(states).context).squeeze(-1)


def plan_trace(state,plan):
    """Unique restricted-growth construction for ascending live member IDs."""
    plan.validate(state.ids('red'),state.ids('targets'),max_members=None)
    ids = sorted(state.ids('red'))
    targets = sorted(state.ids('targets'))
    membership = {i:g for g in plan.groups for i in g.members}
    groups, trace = [], []
    for identity in ids:
        destination = membership.get(identity)
        if destination is None:
            trace.append(len(groups)+len(targets))
        else:
            index = next((j for j,g in enumerate(groups) if g==destination),None)
            if index is None:
                trace.append(len(groups)+targets.index(destination.target))
                groups.append(destination)
            else:
                trace.append(index)
    return trace


class AutoregressiveActor(nn.Module):
    def __init__(self,hidden_dim=128,heads=4,layers=2):
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim,heads=heads,layers=layers)
        self.encoder = EntityEncoder(**self.config)
        self.query = nn.Sequential(nn.Linear(3*hidden_dim,hidden_dim),nn.GELU())
        self.group_key = nn.Sequential(nn.Linear(2*hidden_dim+2,hidden_dim),nn.GELU())
        self.new_key = nn.Linear(hidden_dim,hidden_dim)
        self.reserve_key = nn.Parameter(torch.zeros(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self,states,traces=None,*,deterministic=False):
        if any(not state.ids('red') or not state.ids('targets') for state in states):
            raise ValueError('An action requires live Red and targets')
        enc = self.encoder(states)
        ids = [sorted(s.ids('red')) for s in states]
        targets = [sorted(s.ids('targets')) for s in states]
        groups, reserves = [[] for _ in states],[[] for _ in states]
        generated = [[] for _ in states]
        if traces is not None and (len(traces)!=len(states) or any(len(t)!=len(r) for t,r in zip(traces,ids))):
            raise ValueError('Recorded action trace must cover each live member once')
        batch,h=enc.context.shape
        n,t=max(map(len,ids)),max(map(len,targets))
        device=enc.context.device
        member_indices=torch.tensor([[enc.mappings[b][0,i] for i in roster]+[0]*(n-len(roster))
            for b,roster in enumerate(ids)],device=device)
        target_indices=torch.tensor([[enc.mappings[b][2,i] for i in roster]+[0]*(t-len(roster))
            for b,roster in enumerate(targets)],device=device)
        member_vectors=enc.entities.gather(1,member_indices[...,None].expand(-1,-1,h))
        target_vectors=enc.entities.gather(1,target_indices[...,None].expand(-1,-1,h))
        target_keys=self.new_key(target_vectors)
        sizes=enc.context.new_zeros(batch,n)
        sums=enc.context.new_zeros(batch,n,h)
        assigned_targets=enc.context.new_zeros(batch,n,h)
        reserve_sum=enc.context.new_zeros(batch,h)
        reserve_count=enc.context.new_zeros(batch,1)
        logps=enc.context.new_zeros(batch)
        entropies=enc.context.new_zeros(batch)
        roster_sizes=torch.tensor(list(map(len,ids)),device=device)
        target_sizes=torch.tensor(list(map(len,targets)),device=device)
        for step in range(n):
            active=roster_sizes>step
            group_counts=[len(g) for g in groups]
            ranks=np.zeros((batch,n,1),np.float32)
            for b,old in enumerate(groups):
                canonical=sorted(old)
                for j,group in enumerate(old):
                    ranks[b,j,0]=canonical.index(group)/len(ids[b])
            group_input=torch.cat((sums/sizes.clamp_min(1)[...,None],assigned_targets,
                sizes.log1p()[...,None],torch.as_tensor(ranks,device=device)),-1)
            group_keys=self.group_key(group_input)
            member=member_vectors[:,step]
            query=self.query(torch.cat((enc.context,member,reserve_sum/reserve_count.clamp_min(1)),-1))
            # Fixed tensor slots are masked; external traces retain compact legal-choice indices.
            join_logits=(group_keys*query[:,None]).sum(-1)/math.sqrt(h)+self.bias[0]
            new_logits=(target_keys*query[:,None]).sum(-1)/math.sqrt(h)+self.bias[1]
            reserve_logits=(query*self.reserve_key).sum(-1,keepdim=True)/math.sqrt(h)+self.bias[2]
            join_valid=torch.arange(n,device=device)[None]<torch.tensor(group_counts,device=device)[:,None]
            target_valid=torch.arange(t,device=device)[None]<target_sizes[:,None]
            valid=torch.cat((join_valid,target_valid,torch.ones(batch,1,dtype=torch.bool,device=device)),1)
            logits=torch.cat((join_logits,new_logits,reserve_logits),1).masked_fill(~valid,-1e9)
            dist=Categorical(logits=logits)
            if traces is not None:
                choices=[]
                for b in range(batch):
                    if step>=len(ids[b]):
                        choices.append(n+t)
                        continue
                    token=int(traces[b][step])
                    count=group_counts[b]
                    if not 0<=token<count+len(targets[b])+1:
                        raise ValueError('Recorded action selects an unavailable group')
                    choices.append(token if token<count else n+token-count if token<count+len(targets[b]) else n+t)
                choice=torch.tensor(choices,device=device)
            else:
                choice=logits.argmax(-1) if deterministic else dist.sample()
                choices=choice.detach().cpu().tolist()
            logps=logps+dist.log_prob(choice)*active
            entropies=entropies+dist.entropy()*active
            destination=[]
            for b,fixed in enumerate(choices):
                if step>=len(ids[b]):
                    destination.append(0)
                    continue
                count=group_counts[b]
                token=fixed if fixed<n else count+fixed-n if fixed<n+t else count+len(targets[b])
                generated[b].append(token)
                identity=ids[b][step]
                if fixed<n:
                    old=groups[b][fixed]
                    groups[b][fixed]=Group(old.target,(*old.members,identity))
                    destination.append(fixed)
                elif fixed<n+t:
                    groups[b].append(Group(targets[b][fixed-n],(identity,)))
                    destination.append(count)
                else:
                    reserves[b].append(identity)
                    destination.append(0)
            assigned=(choice<n+t)&active
            created=(choice>=n)&(choice<n+t)&active
            one_hot=F.one_hot(torch.tensor(destination,device=device),n).to(sums.dtype)
            membership=one_hot*assigned[:,None]
            sums=sums+membership[...,None]*member[:,None]
            sizes=sizes+membership
            target_choice=(choice-n).clamp(0,t-1)
            chosen_target=target_vectors.gather(1,target_choice[:,None,None].expand(-1,1,h)).squeeze(1)
            assigned_targets=assigned_targets+(one_hot*created[:,None])[...,None]*chosen_target[:,None]
            reserved=(choice==n+t)&active
            reserve_sum=reserve_sum+reserved[:,None]*member
            reserve_count=reserve_count+reserved[:,None]
        plans = [Grouping(tuple(g),tuple(r)) for g,r in zip(groups,reserves)]
        return dict(plans=plans,traces=generated,log_prob=logps,
            entropy=entropies/roster_sizes,joint_entropy=entropies)


def make_actor(kind,config=None):
    if kind not in ('candidate','autoregressive'):
        raise ValueError(f'Unknown actor type: {kind}')
    return (CandidateActor if kind=='candidate' else AutoregressiveActor)(**(config or {}))


class NeuralPolicy:
    def __init__(self,actor,kind='candidate',candidate_budget=32,proposal_seed=0):
        self.actor,self.kind = actor,kind
        self.candidate_budget,self.proposal_seed = int(candidate_budget),int(proposal_seed)
        self.last_trace = {}
        actor.eval()

    @torch.no_grad()
    def act(self,state):
        if self.kind=='autoregressive':
            result = self.actor([state],deterministic=True)
            plan = result['plans'][0]
            self.last_trace = dict(tokens=result['traces'][0],joint_log_probability=float(result['log_prob'][0]),
                conditional_entropy=float(result['entropy'][0]),joint_entropy=float(result['joint_entropy'][0]),
                candidate_evaluations=0)
            return plan
        # State-derived proposals reproduce training pools and never use real RNG.
        pool = propose_plans(state,self.candidate_budget,self.proposal_seed)
        logits = self.actor([state],[pool])[0]
        action = int(logits.argmax())
        dist = Categorical(logits=logits)
        self.last_trace = dict(candidate_evaluations=len(pool),selected_index=action,
            joint_log_probability=float(dist.log_prob(torch.tensor(action,device=logits.device))),
            conditional_entropy=float(dist.entropy()),joint_entropy=float(dist.entropy()))
        return pool[action]

    __call__ = act


class CoverageProvider:
    """Equal raw proposal count; preserve duplicate samples for coverage accounting."""
    def __init__(self,policy,budget=32,seed=0):
        self.policy,self.budget,self.seed=policy,int(budget),int(seed)

    @torch.no_grad()
    def __call__(self,state):
        if self.policy.kind=='candidate':
            return propose_plans(state,self.budget,self.policy.proposal_seed)
        from .protocol import stable_seed
        devices=[] if next(self.policy.actor.parameters()).device.type=='cpu' else [next(self.policy.actor.parameters()).device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(stable_seed(self.seed,'AR_coverage',state.to_dict()))
            return self.policy.actor([state]*self.budget)['plans']


def load_policy(path,device='cpu'):
    checkpoint = torch.load(Path(path),map_location='cpu',weights_only=False)
    if checkpoint.get('schema')!='v5-neural-checkpoint':
        raise ValueError('Expected a V5 neural policy checkpoint')
    from .protocol import VERSION
    if checkpoint.get('protocol_version',VERSION)!=VERSION:
        raise ValueError('Policy checkpoint uses a different environment protocol')
    actor = make_actor(checkpoint['kind'],checkpoint['model_config'])
    actor.load_state_dict(checkpoint['actor'])
    return NeuralPolicy(actor.to(device),checkpoint['kind'],checkpoint.get('candidate_budget',32),checkpoint.get('proposal_seed',0))


def event_gae(rows,next_values,lam=.95):
    advantages = np.zeros(len(rows),np.float64)
    following = {}
    for i in reversed(range(len(rows))):
        row = rows[i]
        live = float(not row['done'])
        delta = row['reward']+live*float(next_values[i])-row['value']
        advantages[i] = delta+live*lam*following.get(row.get('env',0),0.)
        following[row.get('env',0)] = advantages[i]
    return advantages,advantages+np.asarray([r['value'] for r in rows])


def actor_evaluate(actor,kind,rows):
    states = [r['state'] for r in rows]
    if kind=='autoregressive':
        result = actor(states,[r['trace'] for r in rows])
        return result['log_prob'],result['entropy'],result['joint_entropy']
    logits = actor(states,[r['candidates'] for r in rows])
    dist = Categorical(logits=logits)
    selected = torch.tensor([r['action'] for r in rows],device=logits.device)
    ent = dist.entropy()
    # One categorical candidate choice is one action. Only the AR sequence uses
    # mean conditional entropy; roster-normalized candidate entropy is diagnostic.
    return dist.log_prob(selected),ent,ent


def ppo_update(actor,critic,actor_optimizer,critic_optimizer,rows,kind,config,fraction=0.):
    if set(map(id,actor.parameters())) & set(map(id,critic.parameters())):
        raise ValueError('Actor and critic must own separate parameters')
    batch_size = int(config.get('batch_size',128))
    next_values = np.zeros(len(rows),np.float64)
    with torch.no_grad():
        live = [i for i,r in enumerate(rows) if not r['done']]
        for start in range(0,len(live),batch_size):
            ix = live[start:start+batch_size]
            next_values[ix] = critic([rows[i]['next_state'] for i in ix]).cpu().numpy()
    advantage,returns = event_gae(rows,next_values,float(config.get('gae_lambda',.95)))
    normalized = (advantage-advantage.mean())/max(float(advantage.std()),1e-8)
    device = next(actor.parameters()).device
    old_log = torch.tensor([r['log_prob'] for r in rows],dtype=torch.float32,device=device)
    adv = old_log.new_tensor(normalized)
    target = old_log.new_tensor(returns)
    records,value_losses = [],[]
    rejected_kl = None
    clip = float(config.get('clip',.2))
    maximum = float(config.get('max_gradient_norm',.5))
    entropy_coef = float(config.get('entropy_start',.02))*(1-fraction)+float(config.get('entropy_end',.005))*fraction
    for epoch in range(int(config.get('epochs',4))):
        order = np.random.permutation(len(rows)).tolist()
        for start in range(0,len(rows),batch_size):
            ix = order[start:start+batch_size]
            batch = [rows[i] for i in ix]
            values = critic([r['state'] for r in batch])
            value_loss = F.mse_loss(values,target[ix])
            if not torch.isfinite(value_loss):
                raise FloatingPointError('Nonfinite value loss')
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(),maximum,error_if_nonfinite=True)
            critic_optimizer.step()
            value_losses.append(float(value_loss.detach()))
            if rejected_kl is not None:
                continue
            logp,entropy,joint_entropy = actor_evaluate(actor,kind,batch)
            members=joint_entropy.new_tensor([max(1,len(r['state'].ids('red'))) for r in batch])
            per_member_entropy=joint_entropy/members
            logratio = logp-old_log[ix]
            ratio = logratio.exp()
            kl = float(((ratio-1)-logratio).mean().detach())
            if not math.isfinite(kl):
                raise FloatingPointError('Nonfinite joint policy KL')
            if kl>float(config.get('target_kl',.03)):
                rejected_kl=kl
                continue
            objective = -torch.minimum(ratio*adv[ix],ratio.clamp(1-clip,1+clip)*adv[ix]).mean()
            loss = objective-entropy_coef*entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(actor.parameters(),maximum,error_if_nonfinite=True)
            actor_optimizer.step()
            records.append(dict(actor_loss=float(objective.detach()),actor_total_loss=float(loss.detach()),
                entropy=float(entropy.mean().detach()),per_member_entropy=float(per_member_entropy.mean().detach()),
                joint_entropy=float(joint_entropy.mean().detach()),approx_kl=kl,
                gradient_norm=float(norm),clip_fraction=float(((ratio-1).abs()>clip).float().mean().detach())))
    result = {key:float(np.mean([r[key] for r in records])) for key in records[0]} if records else {}
    return dict(result,value_loss=float(np.mean(value_losses)),actor_updates=len(records),
        critic_updates=len(value_losses),rejected_kl=rejected_kl,entropy_coefficient=entropy_coef,
        rollout_events=len(rows),positive_transitions=sum(r['reward']>0 for r in rows))


def imitation_update(actor,optimizer,rows,kind='candidate',max_gradient_norm=.5):
    if not rows:
        return dict(loss=None,examples=0,optimizer_steps=0)
    if kind=='autoregressive':
        traces = [plan_trace(r['state'],r['teacher']) for r in rows]
        result = actor([r['state'] for r in rows],traces)
        # Joint plan cross entropy, normalized per active member for stable scale.
        sizes = result['log_prob'].new_tensor([len(r['state'].ids('red')) for r in rows])
        loss = -(result['log_prob']/sizes).mean()
    else:
        logits = actor([r['state'] for r in rows],[r['candidates'] for r in rows])
        targets = torch.zeros_like(logits)
        for i,row in enumerate(rows):
            if 'target_distribution' in row:
                target = logits.new_tensor(row['target_distribution'])
                targets[i,:len(target)] = target
            else:
                index = next(j for j,p in enumerate(row['candidates']) if partition_key(p)==partition_key(row['teacher']))
                targets[i,index] = 1.
        loss = -(targets*F.log_softmax(logits,-1)).sum(1).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite imitation loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(actor.parameters(),max_gradient_norm,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()),examples=len(rows),optimizer_steps=1,gradient_norm=float(norm))
