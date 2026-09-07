"""V6 ALMA/AQL and intent MAPPO with physical-event learning semantics.

Only public entities and group relations enter the networks. IDs align members
and define the documented autoregressive traversal; they are never embeddings.
Adaptations follow ALMA (81bd5c4) and marl_mrt2a (b183766); this is an
independent implementation for the fixed HAD executor, not their environments.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from open_score.grouping.domain import Entity, Group, Grouping
from had_env.grouping.rules import responsibilities
from .environment import V6State, potential


def _state(value):
    if isinstance(value, dict) and 'entities' in value:
        entities = [tuple(Entity(int(row[0]), tuple(row[1:4]), tuple(row[4:7]), float(row[7]))
                          for row in rows) for rows in value['entities']]
        step, maximum, initial_r, initial_b, initial_h, initial_k, interval = value['meta']
        return V6State(int(step), int(maximum), 'reactive', *entities, value['previous'],
                       initial_red_count=int(initial_r), initial_blue_count=int(initial_b),
                       initial_blue_health=float(initial_h), initial_target_count=int(initial_k),
                       command_interval=int(interval))
    return V6State.from_dict(value) if isinstance(value, dict) else value


def _pack_state(value):
    state = _state(value)
    return dict(entities=tuple(np.asarray([[e.id, *e.position, *e.velocity, e.health]
                                           for e in getattr(state, side)], dtype=np.float64).reshape(-1, 8)
                               for side in ('red', 'blue', 'targets')),
                meta=(state.step, state.max_steps, state.initial_red_count,
                      state.initial_blue_count, state.initial_blue_health,
                      state.initial_target_count, state.command_interval),
                previous=state.previous)


def _group(value):
    return Grouping.from_dict(value) if isinstance(value, dict) else value


def _transformer(config, layers):
    d = int(config.get('hidden', 128))
    layer = nn.TransformerEncoderLayer(
        d, int(config.get('heads', 4)), int(config.get('ffn', 256)),
        dropout=0., activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, int(layers), norm=nn.LayerNorm(d),
                                 enable_nested_tensor=False)


@dataclass
class EncodedState:
    h: torch.Tensor
    red_ids: tuple
    target_ids: tuple
    red_slice: slice
    blue_slice: slice
    target_slice: slice

    @property
    def reds(self):
        return self.h[self.red_slice]

    @property
    def targets(self):
        return self.h[self.target_slice]


class EntityEncoder(nn.Module):
    """Permutation-equivariant public entity set, plus one context token."""
    feature_size = 35

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.d = int(config.get('hidden', 128))
        self.input = nn.Linear(self.feature_size, self.d)
        self.attention = _transformer(config, config.get('entity_layers', 2))

    @staticmethod
    def features(state):
        state = _state(state)
        reds, blues, targets = (tuple(sorted(state.alive(side), key=lambda e: e.id))
                               for side in ('red', 'blue', 'targets'))
        assignment = state.previous.assignment()
        target_map = {e.id: e for e in targets}
        red_map = {e.id: e for e in reds}
        old_groups = {}
        for group in state.previous.prune(red_map).groups:
            positions = np.asarray([red_map[i].position for i in group.members])
            for identity in group.members:
                old_groups[identity] = (len(group.members), positions.mean(0))
        initial_r = max(int(state.initial_red_count), 1)
        initial_b = max(int(state.initial_blue_count), 1)
        initial_h = max(float(state.initial_blue_health), 1e-8)
        interval = max(int(state.command_interval), 1)
        context = [state.step/max(state.max_steps, 1),
                   (state.max_steps-state.step)/max(state.max_steps, 1),
                   (state.step % interval)/interval,
                   state.initial_red_count/20., state.initial_blue_count/20.,
                   len(reds)/initial_r, len(blues)/initial_b,
                   len(targets)/3., state.initial_target_count/3.,
                   initial_h/20., sum(max(0., e.health) for e in blues)/initial_h,
                   sum(max(0., e.health) for e in targets)/max(len(targets), 1)]
        rows = []
        for kind, entities in enumerate(((None,), reds, blues, targets)):
            for entity in entities:
                row = [float(kind == k) for k in range(4)]
                row += ([0.]*7 if entity is None else
                        [*(np.asarray(entity.position)/5000.),
                         *(np.asarray(entity.velocity)/500.), entity.health])
                row += context
                target = (target_map.get(assignment.get(entity.id))
                          if entity is not None and kind == 1 else None)
                row += ([0.]*7 if target is None else
                        [*(np.asarray(target.position)/5000.),
                         *(np.asarray(target.velocity)/500.), target.health])
                old = old_groups.get(entity.id) if entity is not None and kind == 1 else None
                row += ([0.]*4 if old is None else [old[0]/20., *(old[1]/5000.)])
                row += [float(entity is not None and kind == 1 and
                              assignment.get(entity.id) is None)]
                rows.append(row)
        return np.asarray(rows, np.float32), reds, blues, targets

    def forward(self, state):
        return self.forward_many([state])[0]

    def forward_many(self, states):
        parts = [self.features(state) for state in states]
        device = self.input.weight.device
        rows = [torch.as_tensor(item[0], device=device) for item in parts]
        padded = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        lengths = torch.tensor([len(row) for row in rows], device=device)
        mask = torch.arange(padded.shape[1], device=device)[None] >= lengths[:, None]
        h = self.attention(F.gelu(self.input(padded)), src_key_padding_mask=mask)
        result = []
        for index, (row, reds, blues, targets) in enumerate(parts):
            nr, nb = len(reds), len(blues)
            result.append(EncodedState(h[index, :len(row)], tuple(e.id for e in reds),
                tuple(e.id for e in targets), slice(1, 1+nr), slice(1+nr, 1+nr+nb),
                slice(1+nr+nb, len(row))))
        return result


class GroupQ(nn.Module):
    """Global unbounded Q, with complete old/new partitions and targetwise S2."""
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        d = int(config.get('hidden', 128))
        self.encoder = EntityEncoder(config)
        self.target_fusion = nn.Linear(d+1, d)
        self.group_fusion = nn.Sequential(nn.Linear(3*d+5, d), nn.GELU())
        self.kind = nn.Embedding(5, d)  # context, target, old, new, reserve
        self.relations = _transformer(config, config.get('group_layers', 2))
        self.output = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, state, candidates, probabilities=None):
        return self.forward_many([state], [candidates], [probabilities])[0]

    def forward_many(self, states, candidate_lists, probability_lists=None):
        """Equivalent public pooling, batched over events and all candidates."""
        states = [_state(s) for s in states]
        encodings = self.encoder.forward_many(states)
        device, dtype = encodings[0].h.device, encodings[0].h.dtype
        probability_lists = probability_lists or [None]*len(states)
        entity_length = max(len(enc.h) for enc in encodings)
        target_length = max(len(enc.target_ids) for enc in encodings)
        descriptions, owners, probabilities = [], [], []
        for owner, (state, enc, candidates, probs) in enumerate(zip(states, encodings, candidate_lists, probability_lists)):
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            reds = {e.id: e for e in state.alive('red')}
            rmap = {i: enc.red_slice.start+k for k, i in enumerate(enc.red_ids)}
            bmap = {e.id: enc.blue_slice.start+k for k, e in enumerate(sorted(state.alive('blue'), key=lambda e: e.id))}
            tmap = {i: k for k, i in enumerate(enc.target_ids)}
            old = state.previous.prune(enc.red_ids)
            old_contacts = dict(responsibilities(state, old))
            for candidate_index, candidate in enumerate(candidates):
                candidate = _group(candidate)
                tokens = [(0, (), (), -1, (0.,)*5)]
                tokens.extend((1, (), (), j, (0.,)*5) for j in range(len(enc.target_ids)))
                for partition, kind, contacts in ((old, 2, old_contacts),
                        (candidate, 3, dict(responsibilities(state, candidate)))):
                    for group in partition.groups:
                        members = [i for i in group.members if i in rmap]
                        if not members or group.target not in tmap:
                            continue
                        rs = tuple(rmap[i] for i in members)
                        bs = tuple(bmap[i] for i in contacts.get(group, ()) if i in bmap)
                        center = np.mean([reds[i].position for i in members], 0)/5000.
                        tokens.append((kind, rs, bs, tmap[group.target],
                                       (len(rs)/20., len(bs)/20., *center)))
                    reserve = tuple(rmap[i] for i in partition.reserve if i in rmap)
                    if reserve:
                        tokens.append((kind+3, reserve, (), -1, (0.,)*5))
                descriptions.append(tokens)
                owners.append(owner)
                p = np.zeros(target_length, np.float32)
                if probs is not None:
                    p[:len(enc.target_ids)] = np.asarray(probs[candidate_index])
                probabilities.append(p)
        if not descriptions:
            return [enc.h.new_empty((0,)) for enc in encodings]
        count, length = len(descriptions), max(map(len, descriptions))
        red_weights = np.zeros((count, length, entity_length), np.float32)
        blue_weights = np.zeros_like(red_weights)
        target_index = np.zeros((count, length), np.int64)
        kinds = np.zeros((count, length), np.int64)
        stats = np.zeros((count, length, 5), np.float32)
        valid = np.zeros((count, length), bool)
        for c, tokens in enumerate(descriptions):
            for j, (kind, rs, bs, target, stat) in enumerate(tokens):
                kinds[c, j], stats[c, j], valid[c, j] = kind, stat, True
                if rs:
                    red_weights[c, j, list(rs)] = 1./len(rs)
                if bs:
                    blue_weights[c, j, list(bs)] = 1./len(bs)
                target_index[c, j] = max(target, 0)
        h = nn.utils.rnn.pad_sequence([enc.h for enc in encodings], batch_first=True)
        h = h[torch.as_tensor(owners, device=device)]
        target_vectors = nn.utils.rnn.pad_sequence([enc.targets for enc in encodings], batch_first=True)
        target_vectors = target_vectors[torch.as_tensor(owners, device=device)]
        p = torch.as_tensor(np.asarray(probabilities), dtype=dtype, device=device)
        target_vectors = F.gelu(self.target_fusion(torch.cat([target_vectors, p[:, :, None]], -1)))
        # A terminal with no live targets is not a bootstrap state, but can
        # still be represented for direct interface use.
        if target_length == 0:
            target_vectors = h.new_zeros((count, 1, h.shape[-1]))
        selected_targets = target_vectors.gather(1, torch.as_tensor(target_index, device=device)[:, :, None].expand(-1, -1, h.shape[-1]))
        red = torch.bmm(torch.as_tensor(red_weights, device=device), h)
        blue = torch.bmm(torch.as_tensor(blue_weights, device=device), h)
        group_tokens = self.group_fusion(torch.cat([red, blue, selected_targets,
                            torch.as_tensor(stats, device=device)], -1))
        kinds_t = torch.as_tensor(kinds, device=device)
        base_kind = kinds_t.clamp(max=3)
        base_kind = torch.where(kinds_t >= 5, kinds_t-3, base_kind)
        token = group_tokens+self.kind(base_kind)
        token = torch.where((kinds_t == 0)[:, :, None], h[:, 0, None]+self.kind.weight[0], token)
        token = torch.where((kinds_t == 1)[:, :, None], selected_targets+self.kind.weight[1], token)
        token = torch.where((kinds_t >= 5)[:, :, None], red+self.kind.weight[4]+self.kind(base_kind), token)
        mask = ~torch.as_tensor(valid, device=device)
        encoded = self.relations(token, src_key_padding_mask=mask)
        pooled = (encoded*~mask[:, :, None]).sum(1)/(~mask).sum(1)[:, None]
        values = self.output(pooled).squeeze(-1)
        return list(values.split([len(candidates) for candidates in candidate_lists]))


def _pair_features(state, identity, target_ids, device):
    red = next(e for e in state.alive('red') if e.id == identity)
    targets = {e.id: e for e in state.alive('targets')}
    old = state.previous.assignment().get(identity)
    rows = []
    for identity in target_ids:
        if identity is None:
            rows.append([0.]*7+[float(old is None)])
        else:
            entity = targets[identity]
            displacement = np.asarray(entity.position)-red.position
            rows.append([*(displacement/5000.),
                         *((np.asarray(entity.velocity)-red.velocity)/500.),
                         np.linalg.norm(displacement)/5000., float(old == identity)])
    return torch.as_tensor(rows, dtype=torch.float32, device=device)


def _draw(logits, rng, forced=None):
    log_probs = F.log_softmax(logits, -1)
    probs = log_probs.exp()
    entropy = -(probs*log_probs.masked_fill(~torch.isfinite(log_probs), 0.)).sum(-1)
    if forced is None:
        raw = probs.detach().cpu().numpy()
        choices = [int(rng.choice(len(row), p=row.astype(float)/row.astype(float).sum())) for row in raw]
        choice = torch.tensor(choices, dtype=torch.long, device=logits.device)
    else:
        choice = torch.as_tensor(forced, dtype=torch.long, device=logits.device)
    return choice, log_probs.gather(-1, choice[:, None]).squeeze(-1), entropy


class PointerProposal(nn.Module):
    def __init__(self, config=None, grouping=True):
        super().__init__()
        config = config or {}
        d = int(config.get('hidden', 128))
        self.grouping = bool(grouping)
        self.encoder = EntityEncoder(config)
        self.reserve = nn.Parameter(torch.zeros(d))
        self.new = nn.Parameter(torch.zeros(d))
        self.target_pointer = nn.Sequential(nn.Linear(2*d+8, d), nn.GELU(), nn.Linear(d, 1))
        self.target_update = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d))
        self.group_pointer = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, state, count=1, rng=None, actions=None):
        return self.forward_many([state], [count], rng=rng,
                                 action_lists=None if actions is None else [actions])[0]

    def forward_many(self, states, counts=None, rng=None, action_lists=None):
        """Pointer decoding batches all event/candidate rows at each member."""
        states = [_state(s) for s in states]
        rng = np.random.default_rng() if rng is None else rng
        counts = [len(actions) for actions in action_lists] if action_lists is not None else counts
        encodings = self.encoder.forward_many(states)
        device = encodings[0].h.device
        owners = [i for i, count in enumerate(counts) for _ in range(count)]
        owner_t = torch.as_tensor(owners, device=device)
        n, kmax = max(len(e.red_ids) for e in encodings), max(len(e.target_ids) for e in encodings)
        b, d = len(owners), self.reserve.numel()
        targets = nn.utils.rnn.pad_sequence([e.targets for e in encodings], batch_first=True)
        targets = torch.cat([targets, self.reserve[None, None].expand(len(states), 1, -1)], 1)
        options = targets[owner_t]
        reds = nn.utils.rnn.pad_sequence([e.reds for e in encodings], batch_first=True)[owner_t]
        count_red = np.asarray([len(encodings[o].red_ids) for o in owners])
        count_target = np.asarray([len(encodings[o].target_ids) for o in owners])
        target_mask = np.arange(kmax+1)[None] < count_target[:, None]
        target_mask[:, -1] = True
        pairs = np.zeros((len(states), n, kmax+1, 8), np.float32)
        for owner, (state, enc) in enumerate(zip(states, encodings)):
            for member, identity in enumerate(enc.red_ids):
                p = _pair_features(state, identity, (*enc.target_ids, None), 'cpu').numpy()
                pairs[owner, member, :len(enc.target_ids)] = p[:-1]
                pairs[owner, member, -1] = p[-1]
        pair_tensor = torch.as_tensor(pairs, device=device)[owner_t]
        assignment = np.full((b, n), kmax, np.int64)
        actions = ([_group(g) for items in action_lists for g in items]
                   if action_lists is not None else None)
        forced_assignment = None
        if actions is not None:
            forced_assignment = assignment.copy()
            for row, (owner, action) in enumerate(zip(owners, actions)):
                enc = encodings[owner]
                desired = action.assignment()
                for j, identity in enumerate(enc.red_ids):
                    t = desired[identity]
                    forced_assignment[row, j] = kmax if t is None else enc.target_ids.index(t)
        logp = options.new_zeros(b)
        entropy = options.new_zeros(b)
        tokens = count_red.copy()
        for member in range(n):
            active = count_red > member
            mask = target_mask.copy()
            mask[~active, :-1] = False
            vector = reds[:, member]
            logits = self.target_pointer(torch.cat([vector[:, None].expand(-1, kmax+1, -1),
                                                    options, pair_tensor[:, member]], -1)).squeeze(-1)
            logits = logits.masked_fill(~torch.as_tensor(mask, device=device), -torch.inf)
            forced = None if forced_assignment is None else forced_assignment[:, member]
            choice, lp, ent = _draw(logits, rng, forced)
            assignment[:, member] = choice.detach().cpu().numpy()
            live = torch.as_tensor(active, device=device)
            logp, entropy = logp+lp*live, entropy+ent*live
            selected = options[torch.arange(b, device=device), choice]
            update = self.target_update(torch.cat([selected, vector], -1))
            options = options+F.one_hot(choice, kmax+1)[:, :, None]*update[:, None]*live[:, None, None]
        partitions = [[] for _ in range(b)]
        desired_groups = ([{i: frozenset(g.members) for g in action.groups for i in g.members}
                           for action in actions] if actions is not None else None)
        if self.grouping:
            for member in range(n):
                rows, legals = [], []
                for row, owner in enumerate(owners):
                    if member >= count_red[row] or assignment[row, member] == kmax:
                        continue
                    tokens[row] += 1
                    target = int(assignment[row, member])
                    legal = [i for i, (t, _) in enumerate(partitions[row]) if t == target]
                    if not legal:
                        partitions[row].append((target, [member]))
                    else:
                        rows.append(row)
                        legals.append(legal)
                if not rows:
                    continue
                maximum = max(map(len, legals))+1
                weights = np.zeros((len(rows), maximum, n), np.float32)
                valid = np.zeros((len(rows), maximum), bool)
                new_mask = np.zeros_like(valid)
                forced = [] if actions is not None else None
                for index, (row, legal) in enumerate(zip(rows, legals)):
                    valid[index, :len(legal)+1] = True
                    new_mask[index, len(legal)] = True
                    for j, group_index in enumerate(legal):
                        members = partitions[row][group_index][1]
                        weights[index, j, members] = 1./len(members)
                    if actions is not None:
                        enc = encodings[owners[row]]
                        desired = desired_groups[row][enc.red_ids[member]]
                        forced.append(next((j for j, group_index in enumerate(legal)
                            if enc.red_ids[partitions[row][group_index][1][0]] in desired), len(legal)))
                row_t = torch.as_tensor(rows, device=device)
                vectors = torch.bmm(torch.as_tensor(weights, device=device), reds[row_t])
                vectors = vectors+torch.as_tensor(new_mask, device=device)[:, :, None]*self.new
                logits = self.group_pointer(torch.cat([reds[row_t, member, None].expand(-1, maximum, -1), vectors], -1)).squeeze(-1)
                logits = logits.masked_fill(~torch.as_tensor(valid, device=device), -torch.inf)
                choice, lp, ent = _draw(logits, rng, forced)
                logp = logp.index_add(0, row_t, lp)
                entropy = entropy.index_add(0, row_t, ent)
                for row, legal, selected in zip(rows, legals, choice.detach().cpu().tolist()):
                    if selected == len(legal):
                        partitions[row].append((int(assignment[row, member]), [member]))
                    else:
                        partitions[row][legal[selected]][1].append(member)
        else:
            for row in range(b):
                partitions[row] = [(target, [j for j in range(count_red[row]) if assignment[row, j] == target])
                                   for target in range(count_target[row])
                                   if target in assignment[row, :count_red[row]]]
        result = []
        for row, owner in enumerate(owners):
            enc = encodings[owner]
            result.append(Grouping(tuple(Group(enc.target_ids[t], tuple(enc.red_ids[j] for j in members))
                                         for t, members in partitions[row]),
                                   tuple(enc.red_ids[j] for j in range(count_red[row]) if assignment[row, j] == kmax)))
        splits, start = [], 0
        for count in counts:
            splits.append((result[start:start+count], logp[start:start+count],
                           entropy[start:start+count], tokens[start:start+count]))
            start += count
        return splits


class IntentActor(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        d = int(config.get('hidden', 128))
        self.encoder = EntityEncoder(config)
        self.reserve = nn.Parameter(torch.zeros(d))
        self.pointer = nn.Sequential(nn.Linear(2*d+8, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, state):
        state = _state(state)
        enc = self.encoder(state)
        options = torch.cat([enc.targets, self.reserve[None]], 0)
        if not enc.red_ids:
            return enc.h.new_empty((0, len(options))), enc.red_ids, (*enc.target_ids, None)
        pair = torch.stack([_pair_features(state, i, (*enc.target_ids, None), enc.h.device)
                            for i in enc.red_ids])
        logits = self.pointer(torch.cat([enc.reds[:, None].expand(-1, len(options), -1),
                                         options[None].expand(len(enc.red_ids), -1, -1), pair], -1)).squeeze(-1)
        return logits, enc.red_ids, (*enc.target_ids, None)


class TeamCritic(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        d = int(config.get('hidden', 128))
        self.encoder = EntityEncoder(config)
        self.output = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, state):
        return self.output(self.encoder(state).h.mean(0)).squeeze(-1)


def _compact_episode(episode):
    result = []
    for transition in episode:
        row = {key: transition[key] for key in
               ('reward', 'native_reward', 'done', 'delta', 'old_logp', 'policy_version')
               if key in transition}
        row['state'] = _pack_state(transition['state'])
        row['next_state'] = _pack_state(transition['next_state'])
        row['action'] = _group(transition['action'])
        result.append(row)
    if not result or not result[-1]['done'] or any(r['done'] for r in result[:-1]):
        raise ValueError('Replay accepts complete episodes ending at a genuine terminal only')
    return result


def _epsilon(config, physical_steps):
    fraction = min(1., max(0., physical_steps)/max(1, config['decay_physical_steps']))
    return float(config['start']+(config['end']-config['start'])*fraction)


def _explore(state, grouping, epsilon, rng, allow_groups):
    """Perturb the selected candidate, retaining its surviving partition."""
    if epsilon <= 0 or not state.ids('red'):
        return grouping
    assignment = grouping.assignment()
    target_ids = (*tuple(sorted(state.ids('targets'))), None)
    for identity in sorted(assignment):
        if rng.random() < epsilon:
            assignment[identity] = target_ids[int(rng.integers(len(target_ids)))]
    if not allow_groups:
        return Grouping(tuple(Group(t, tuple(i for i in sorted(assignment) if assignment[i] == t))
                              for t in target_ids[:-1] if t in assignment.values()),
                        tuple(i for i in assignment if assignment[i] is None))
    groups = [(g.target, [i for i in g.members if assignment[i] == g.target]) for g in grouping.groups]
    groups = [(t, members) for t, members in groups if members]
    used = {i for _, members in groups for i in members}
    for identity in sorted(assignment):
        if assignment[identity] is not None and identity not in used:
            groups.append((assignment[identity], [identity]))
    for identity in sorted(assignment):
        target = assignment[identity]
        if target is None or rng.random() >= epsilon:
            continue
        groups = [(t, [i for i in members if i != identity]) for t, members in groups]
        groups = [(t, members) for t, members in groups if members]
        legal = [k for k, (t, _) in enumerate(groups) if t == target]
        selected = int(rng.integers(len(legal)+1))
        if selected == len(legal):
            groups.append((target, [identity]))
        else:
            groups[legal[selected]][1].append(identity)
    return Grouping(tuple(Group(t, tuple(m)) for t, m in groups),
                    tuple(i for i in assignment if assignment[i] is None))


class AQLLearner:
    def __init__(self, method, config, seed, device='cpu', s2=None):
        self.method, self.config, self.device = method, config, torch.device(device)
        self.settings = config.get('aql', {})
        self.rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))
        self.proposal = PointerProposal(config.get('model'), grouping=method != 'ALMA_Alloc').to(self.device)
        self.q = GroupQ(config.get('model')).to(self.device)
        self.target = copy.deepcopy(self.q).requires_grad_(False)
        self.s2 = s2
        if method == 'ALMA_S2' and s2 is None:
            raise ValueError('ALMA_S2 requires a frozen, trained S2 scorer')
        settings = self.settings.get('optimizer', {})
        options = dict(lr=settings.get('learning_rate', 5e-4),
                       alpha=settings.get('alpha', .99), eps=settings.get('epsilon', 1e-5),
                       weight_decay=settings.get('weight_decay', 0.), momentum=0., centered=False)
        self.q_optimizer = torch.optim.RMSprop(self.q.parameters(), **options)
        self.proposal_optimizer = torch.optim.RMSprop(self.proposal.parameters(), **options)
        self.clip = float(settings.get('gradient_clip', 10.))
        capacity = self.settings.get('replay', {}).get('capacity_complete_episodes', 5000)
        self.replay = deque(maxlen=int(capacity))
        self.total_episodes = self.total_events = self.credit = self.updates = self.proposal_updates = 0
        self.target_copies = self.last_target_threshold = 0
        self.warm = False
        self.policy_version = 0

    def _probabilities(self, state, candidates):
        if self.method != 'ALMA_S2':
            return None, 0
        values = np.asarray(self.s2.probabilities(state, candidates), np.float32)
        return values, int(getattr(self.s2, 'last_score_rows', values.size))

    @torch.no_grad()
    def _select(self, state, count):
        return self._select_many([state], count)[0]

    @torch.no_grad()
    def _select_many(self, states, count):
        proposals = self.proposal.forward_many(states, [count]*len(states), rng=self.rng)
        unique = [list(dict.fromkeys(row[0])) for row in proposals]
        outputs = [self._probabilities(s, plans) for s, plans in zip(states, unique)]
        values = self.q.forward_many(states, unique, [row[0] for row in outputs])
        result = []
        for (candidates, _, _, tokens), plans, (probabilities, rows), scores in zip(proposals, unique, outputs, values):
            best = int(scores.argmax().item())
            result.append((candidates, plans[best], float(scores[best]), probabilities, plans,
                dict(candidate_generated=len(candidates), candidate_scored=len(plans),
                     internal_tokens=int(tokens.sum()), s2_rows=rows,
                     candidate_duplicate_fraction=1.-len(plans)/len(candidates))))
        return result

    @torch.no_grad()
    def act(self, state, physical_steps=0, explore=True, rng=None):
        state = _state(state)
        external_rng = self.rng
        if rng is not None:
            self.rng = rng
        try:
            count = int(self.settings.get('candidates', {}).get('current' if explore else 'evaluation', 32))
            candidates, selected, value, _, _, metrics = self._select(state, count)
            exploration = self.settings.get('exploration', {})
            member = _epsilon(exploration.get('member', dict(start=1., end=0., decay_physical_steps=250000)), physical_steps) if explore else 0.
            candidate = _epsilon(exploration.get('candidate', dict(start=1., end=.05, decay_physical_steps=1000000)), physical_steps) if explore else 0.
            if explore and self.rng.random() < candidate:
                selected = candidates[int(self.rng.integers(len(candidates)))]
            result = _explore(state, selected, member, self.rng, self.method != 'ALMA_Alloc')
            result.validate(state.ids('red'), state.ids('targets'), max_members=None)
            metrics.update(member_epsilon=member, candidate_epsilon=candidate,
                           greedy_candidate_q=value, greedy_q_native=value+.5*potential(state),
                           policy_version=self.policy_version)
            return result, metrics
        finally:
            self.rng = external_rng

    def _update(self, batch):
        prepared = []
        metrics = dict(candidate_generated=0, candidate_scored=0, internal_tokens=0, s2_rows=0)
        microbatch = int(self.config.get('resources', {}).get('event_microbatch', 8))
        with torch.no_grad():
            for start in range(0, len(batch), microbatch):
                rows = batch[start:start+microbatch]
                states = [_state(row['state']) for row in rows]
                selections = self._select_many(states, int(self.settings.get('candidates', {}).get('current', 32)))
                live_rows = [i for i, row in enumerate(rows) if not row['done']]
                next_states = [_state(rows[i]['next_state']) for i in live_rows]
                targets = [float(row['reward']) for row in rows]
                if next_states:
                    next_selected = self._select_many(next_states, int(self.settings.get('candidates', {}).get('next', 32)))
                    next_probs = [None if row[3] is None else row[3][row[4].index(row[1]):row[4].index(row[1])+1]
                                  for row in next_selected]
                    next_values = self.target.forward_many(next_states, [[row[1]] for row in next_selected], next_probs)
                    for i, selection, value in zip(live_rows, next_selected, next_values):
                        targets[i] += float(value[0])
                        for key in metrics:
                            metrics[key] += selection[-1][key]
                for row, state, selection, target_value in zip(rows, states, selections, targets):
                    candidates, best, _, _, _, info = selection
                    for key in metrics:
                        metrics[key] += info[key]
                    historical = _group(row['action'])
                    historical_probs, scoring_rows = self._probabilities(state, [historical])
                    metrics['s2_rows'] += scoring_rows
                    prepared.append((state, historical, historical_probs, target_value, candidates, best))
        # Every target and proposal teacher was fixed before this Q step.
        self.q_optimizer.zero_grad(set_to_none=True)
        squared_error = q_total = target_total = 0.
        for start in range(0, len(prepared), microbatch):
            rows = prepared[start:start+microbatch]
            predictions = self.q.forward_many([r[0] for r in rows], [[r[1]] for r in rows], [r[2] for r in rows])
            prediction = torch.cat(predictions)
            targets = prediction.new_tensor([r[3] for r in rows])
            loss = (prediction-targets).square().sum()
            (loss/len(prepared)).backward()
            squared_error += float(loss.detach())
            q_total += float(prediction.detach().sum())
            target_total += float(targets.sum())
        q_norm = float(nn.utils.clip_grad_norm_(self.q.parameters(), self.clip))
        self.q_optimizer.step()
        self.updates += 1
        valid = [row for row in prepared if row[0].ids('red')]
        proposal_loss = entropy_total = nll_total = 0.
        if valid:
            self.proposal_optimizer.zero_grad(set_to_none=True)
            entropy_coefficient = float(self.settings.get('proposal_loss', {}).get('entropy_coefficient', .01))
            for start in range(0, len(valid), microbatch):
                rows = valid[start:start+microbatch]
                outputs = self.proposal.forward_many([r[0] for r in rows],
                    rng=self.rng, action_lists=[r[4] for r in rows])
                losses = []
                for (state, _, _, _, candidates, best), (_, logp, entropy, tokens) in zip(rows, outputs):
                    metrics['internal_tokens'] += int(tokens.sum())
                    selected = candidates.index(best)
                    n = len(state.ids('red'))
                    loss = (-logp[selected]-entropy_coefficient*entropy.mean())/n
                    losses.append(loss)
                    proposal_loss += float(loss.detach())
                    entropy_total += float(entropy.mean().detach())/n
                    nll_total += float(-logp[selected].detach())/n
                (torch.stack(losses).sum()/len(valid)).backward()
            proposal_norm = float(nn.utils.clip_grad_norm_(self.proposal.parameters(), self.clip))
            self.proposal_optimizer.step()
            self.proposal_updates += 1
        else:
            proposal_norm = 0.
        self.policy_version += 1
        metrics.update(q_loss=squared_error/len(prepared), td_rmse=(squared_error/len(prepared))**.5,
                       q_mean=q_total/len(prepared), td_target_mean=target_total/len(prepared),
                       proposal_loss=proposal_loss/max(len(valid), 1),
                       proposal_nll_per_member=nll_total/max(len(valid), 1),
                       proposal_entropy_per_member=entropy_total/max(len(valid), 1),
                       q_gradient_norm=q_norm, proposal_gradient_norm=proposal_norm)
        return metrics

    def learn(self, episodes):
        warmup = self.settings.get('warmup', {})
        new_events = 0
        for episode in episodes:
            rows = _compact_episode(episode)
            self.replay.append(rows)
            self.total_episodes += 1
            self.total_events += len(rows)
            new_events += len(rows)
            if self.warm:
                self.credit += len(rows)
            elif (self.total_episodes >= int(warmup.get('complete_episodes', 32)) and
                  sum(map(len, self.replay)) >= int(warmup.get('valid_events', 64))):
                self.warm = True
        result, count = {}, 0
        interval = int(self.settings.get('events_per_update', 16))
        events = [row for episode in self.replay for row in episode]
        batch_size = int(self.settings.get('batch_events', 64))
        while self.warm and self.credit >= interval:
            indices = self.rng.choice(len(events), size=batch_size, replace=False)
            metrics = self._update([events[int(i)] for i in indices])
            for key, value in metrics.items():
                result[key] = result.get(key, 0.)+value
            count += 1
            self.credit -= interval
        additive = {'candidate_generated', 'candidate_scored', 'internal_tokens', 's2_rows'}
        result = {key: value if key in additive else value/max(count, 1) for key, value in result.items()}
        threshold = self.total_episodes//int(self.settings.get('target_update', {}).get('new_episode_interval', 50))
        if threshold > self.last_target_threshold:
            self.target.load_state_dict(self.q.state_dict())
            self.last_target_threshold = threshold
            self.target_copies += 1
        positives = sum(sum(float(row.get('native_reward', 0.)) for row in ep) > 0 for ep in self.replay)
        result.update(q_optimizer_steps=self.updates, proposal_optimizer_steps=self.proposal_updates,
                      optimizer_steps=self.updates+self.proposal_updates, block_updates=count,
                      replay_episodes=len(self.replay), replay_events=len(events), replay_positive_episodes=positives,
                      new_events=new_events, update_event_remainder=self.credit, target_copies=self.target_copies)
        return result

    def policy_state_dict(self):
        return dict(method=self.method, proposal=self.proposal.state_dict(), q=self.q.state_dict(),
                    policy_version=self.policy_version)

    def load_policy_state_dict(self, payload):
        if payload['method'] != self.method:
            raise ValueError('Policy method mismatch')
        self.proposal.load_state_dict(payload['proposal'])
        self.q.load_state_dict(payload['q'])
        self.policy_version = int(payload.get('policy_version', 0))

    def state_dict(self, include_replay=True):
        return {**self.policy_state_dict(), 'target': self.target.state_dict(),
                'q_optimizer': self.q_optimizer.state_dict(),
                'proposal_optimizer': self.proposal_optimizer.state_dict(),
                'rng': copy.deepcopy(self.rng.bit_generator.state),
                'counters': {key: getattr(self, key) for key in
                             ('total_episodes', 'total_events', 'credit', 'updates', 'proposal_updates',
                              'target_copies', 'last_target_threshold', 'warm')},
                'replay': list(self.replay) if include_replay else None}

    def load_state_dict(self, payload):
        self.load_policy_state_dict(payload)
        self.target.load_state_dict(payload['target'])
        self.q_optimizer.load_state_dict(payload['q_optimizer'])
        self.proposal_optimizer.load_state_dict(payload['proposal_optimizer'])
        self.rng.bit_generator.state = payload['rng']
        for key, value in payload['counters'].items():
            setattr(self, key, value)
        if payload.get('replay') is not None:
            self.replay.clear()
            self.replay.extend(payload['replay'])


class MAPPOLearner:
    def __init__(self, method, config, seed, device='cpu', s2=None):
        self.method, self.config, self.device = method, config, torch.device(device)
        self.settings = config.get('mappo', {})
        self.rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))
        self.actor = IntentActor(config.get('model')).to(self.device)
        self.critic = TeamCritic(config.get('model')).to(self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        settings = self.settings.get('optimizer', {})
        options = dict(lr=settings.get('learning_rate', 5e-4), betas=tuple(settings.get('betas', [.9, .999])),
                       eps=settings.get('epsilon', 1e-8), weight_decay=settings.get('weight_decay', 0.))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), **options)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), **options)
        self.clip = float(settings.get('gradient_clip', 10.))
        self.pending = []
        self.batches = self.actor_steps = self.critic_steps = self.policy_version = self.total_episodes = 0

    @torch.no_grad()
    def act(self, state, physical_steps=0, explore=True, rng=None):
        state = _state(state)
        logits, identities, targets = self.actor(state)
        rng = self.rng if rng is None else rng
        if identities:
            choices, logp, entropy = _draw(logits, rng)
            assignments = {i: targets[c] for i, c in zip(identities, choices.cpu().tolist())}
            old_logp = {str(i): float(v) for i, v in zip(identities, logp.cpu().tolist())}
            ent = float(entropy.mean())
        else:
            assignments, old_logp, ent = {}, {}, 0.
        grouping = Grouping(tuple(Group(t, tuple(i for i in identities if assignments[i] == t))
                                  for t in targets[:-1] if t in assignments.values()),
                            tuple(i for i in identities if assignments[i] is None))
        return grouping, dict(old_logp=old_logp, policy_version=self.policy_version,
                              actor_entropy=ent, internal_tokens=len(identities), candidate_scored=1,
                              s2_rows=0, value=float(self.critic(state)))

    def learn(self, episodes):
        for episode in episodes:
            rows = _compact_episode(episode)
            for row in rows:
                if int(row.get('policy_version', self.policy_version)) != self.policy_version:
                    raise ValueError('MAPPO batch must contain ten fresh episodes from one actor version')
            self.pending.append(rows)
            self.total_episodes += 1
        batch_size = int(self.settings.get('episodes_per_batch', 10))
        if len(self.pending) < batch_size:
            return dict(pending_episodes=len(self.pending), optimizer_steps=self.actor_steps+self.critic_steps)
        if len(self.pending) != batch_size:
            raise ValueError('Collect exactly ten complete MAPPO episodes before changing actor version')
        prepared = []
        nstep = int(self.settings.get('n_step_events', 5))
        with torch.no_grad():
            for episode in self.pending:
                for index, row in enumerate(episode):
                    stop = min(index+nstep, len(episode))
                    target = sum(float(episode[j]['reward']) for j in range(index, stop))
                    if stop < len(episode):
                        target += float(self.target(_state(episode[stop]['state'])))
                    prepared.append((_state(row['state']), _group(row['action']), row.get('old_logp', {}), target))
        member_count = sum(len(state.ids('red')) for state, _, _, _ in prepared)
        metrics = dict(actor_loss=0., value_loss=0., actor_entropy=0., value_mean=0.)
        epochs = int(self.settings.get('epochs', 4))
        for _ in range(epochs):
            with torch.no_grad():
                advantages = [target-float(self.critic(state)) for state, _, _, target in prepared]
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            for (state, grouping, old, target), advantage in zip(prepared, advantages):
                value = self.critic(state)
                loss = (value-target).square()
                (loss/len(prepared)).backward()
                metrics['value_loss'] += float(loss.detach())/(len(prepared)*epochs)
                metrics['value_mean'] += float(value.detach())/(len(prepared)*epochs)
                if not state.ids('red'):
                    continue
                logits, identities, targets = self.actor(state)
                actions = [targets.index(grouping.assignment()[i]) for i in identities]
                _, logp, entropy = _draw(logits, self.rng, actions)
                if isinstance(old, dict):
                    old_values = [old[str(i)] if str(i) in old else old[i] for i in identities]
                else:
                    old_values = old
                old_values = torch.as_tensor(old_values, device=self.device, dtype=torch.float32)
                ratio = (logp-old_values).exp()
                clip = float(self.settings.get('clip', .2))
                surrogate = torch.minimum(ratio*advantage, ratio.clamp(1.-clip, 1.+clip)*advantage)
                actor_loss = (-surrogate-float(self.settings.get('entropy_coefficient', .01))*entropy).sum()/max(member_count, 1)
                actor_loss.backward()
                metrics['actor_loss'] += float(actor_loss.detach())/epochs
                metrics['actor_entropy'] += float(entropy.sum().detach())/(max(member_count, 1)*epochs)
            if member_count:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
                self.actor_optimizer.step()
                self.actor_steps += 1
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip)
            self.critic_optimizer.step()
            self.critic_steps += 1
        self.pending.clear()
        self.batches += 1
        self.policy_version += 1
        if self.batches % int(self.settings.get('target_update_batches', 200)) == 0:
            self.target.load_state_dict(self.critic.state_dict())
        metrics.update(actor_optimizer_steps=self.actor_steps, critic_optimizer_steps=self.critic_steps,
                       optimizer_steps=self.actor_steps+self.critic_steps, update_batches=self.batches,
                       batch_events=len(prepared), batch_alive_entries=member_count, pending_episodes=0)
        return metrics

    def policy_state_dict(self):
        return dict(method=self.method, actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    policy_version=self.policy_version)

    def load_policy_state_dict(self, payload):
        if payload['method'] != self.method:
            raise ValueError('Policy method mismatch')
        self.actor.load_state_dict(payload['actor'])
        self.critic.load_state_dict(payload['critic'])
        self.policy_version = int(payload.get('policy_version', 0))

    def state_dict(self, include_replay=True):
        return {**self.policy_state_dict(), 'target': self.target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'rng': copy.deepcopy(self.rng.bit_generator.state),
                'pending': self.pending if include_replay else [],
                'counters': {key: getattr(self, key) for key in
                             ('batches', 'actor_steps', 'critic_steps', 'total_episodes')}}

    def load_state_dict(self, payload):
        self.load_policy_state_dict(payload)
        self.target.load_state_dict(payload['target'])
        self.actor_optimizer.load_state_dict(payload['actor_optimizer'])
        self.critic_optimizer.load_state_dict(payload['critic_optimizer'])
        self.rng.bit_generator.state = payload['rng']
        self.pending = list(payload.get('pending', []))
        for key, value in payload['counters'].items():
            setattr(self, key, value)


def make_learner(method, config, seed, device='cpu', s2=None):
    if method == 'MAPPO_Intent':
        return MAPPOLearner(method, config, seed, device, s2)
    if method in ('ALMA_Alloc', 'ALMA_Group', 'ALMA_S2'):
        return AQLLearner(method, config, seed, device, s2)
    raise ValueError(f'{method} is not a learned v6 method')
