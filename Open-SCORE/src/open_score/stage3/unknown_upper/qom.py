"""Public-trajectory VQ opponent model with dynamic entity/target decoding.

Offline labels enter losses only. Online inference takes PublicState and a
public recurrent history. Code indices are learned modes, not named policies.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .domain import PublicState


def entity_array(entities):
    return np.asarray([[*(np.asarray(x.position)/2500), *(np.asarray(x.velocity)/300), x.health] for x in entities], dtype=np.float32).reshape(-1, 7)


def state_arrays(state):
    red, blue, targets = [state.alive(side) if side != "targets" else state.targets for side in ["red", "blue", "targets"]]
    b, t = entity_array(blue), entity_array(targets)
    pair = np.zeros((len(blue), len(targets), 13), np.float32)
    for i, entity in enumerate(blue):
        for j, target in enumerate(targets):
            relative = np.asarray(target.position)-entity.position
            heading = float(np.dot(relative, entity.velocity)/(max(1., np.linalg.norm(relative)*np.linalg.norm(entity.velocity))))
            distances = [np.linalg.norm(np.asarray(x.position)-target.position) for x in red]
            pair[i,j] = [*(relative/5000), *b[i,3:6], entity.health, target.health,
                         heading, sum(d<1000 for d in distances)/30, min(distances,default=5000)/5000,
                         state.red_history_counts[j]/max(1,30*state.max_steps), state.step/state.max_steps]
    context = np.asarray([state.step/state.max_steps, float(state.lower == "split_rush")],np.float32)
    return entity_array(red), b, t, pair, context


class QuantizedOpponentModel(nn.Module):
    def __init__(self, codes=8, latent=32):
        super().__init__()
        self.codes, self.latent = codes, latent
        self.entity = nn.Sequential(nn.Linear(7,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU())
        self.temporal = nn.GRU(98, latent, batch_first=True)
        self.codebook = nn.Embedding(codes, latent)
        nn.init.uniform_(self.codebook.weight, -.15, .15)
        self.target_head = nn.Sequential(nn.Linear(13+latent*2+2,64),nn.ReLU(),nn.Linear(64,1))
        self.group_head = nn.Sequential(nn.Linear(14+latent*2+2,64),nn.ReLU(),nn.Linear(64,1))

    def encode(self, red, blue, targets, red_mask, blue_mask, target_mask, context, hidden=None):
        def pool(x, mask):
            embeddings = self.entity(x)
            return (embeddings*mask[...,None]).sum(-2)/mask.sum(-1,keepdim=True).clamp_min(1)
        global_state = torch.cat([pool(red,red_mask),pool(blue,blue_mask),pool(targets,target_mask),context],dim=-1)
        return self.temporal(global_state,hidden)

    def decode(self, blue, pair, context, history, code, target_mask):
        shape = pair.shape[:-1]
        h = history[...,None,None,:].expand(*shape,self.latent)
        z = code[:,None,None,None,:].expand(*shape,self.latent)
        c = context[...,None,None,:].expand(*shape,2)
        target_logits = self.target_head(torch.cat([pair,h,z,c],-1)).squeeze(-1)
        target_logits = target_logits.masked_fill(~target_mask[...,None,:], -1e9)
        n = blue.shape[-2]
        left = blue[..., :,None,:].expand(*blue.shape[:-2],n,n,7)
        right = blue[...,None,:,:].expand_as(left)
        # Symmetric relation representation (mean and absolute difference).
        features = torch.cat([(left+right)/2,torch.abs(left-right)],-1)
        pair_shape = features.shape[:-1]
        gh = history[...,None,None,:].expand(*pair_shape,self.latent)
        gz = code[:,None,None,None,:].expand(*pair_shape,self.latent)
        gc = context[...,None,None,:].expand(*pair_shape,2)
        groups = self.group_head(torch.cat([features,gh,gz,gc],-1)).squeeze(-1)
        return target_logits, groups

    def forward(self, batch):
        h,_ = self.encode(*(batch[k] for k in ["red","blue","targets","red_mask","blue_mask","target_mask","context"]))
        last = h[torch.arange(len(h),device=h.device),batch["lengths"]-1]
        distance = ((last[:,None,:]-self.codebook.weight[None,:,:])**2).sum(-1)
        index = distance.argmin(-1)
        code = self.codebook(index)
        quantized = last+(code-last).detach()
        logits, groups = self.decode(batch["blue"], batch["pair"],batch["context"],h,quantized,batch["target_mask"])
        vq = F.mse_loss(code,last.detach())+.25*F.mse_loss(last,code.detach())
        return logits,groups,vq,index,last


def collate(episodes, device):
    sequences = [[(PublicState.from_dict(frame["public"]), frame["blue_action"]) for frame in ep["commands"]] for ep in episodes]
    batch_size, length = len(sequences),max(map(len,sequences))
    nr = max(len(s.red) for seq in sequences for s,_ in seq)
    nb = max(len(s.blue) for seq in sequences for s,_ in seq)
    nt = max(len(s.targets) for seq in sequences for s,_ in seq)
    arrays = {"red":np.zeros((batch_size,length,nr,7),np.float32), "blue":np.zeros((batch_size,length,nb,7),np.float32),
              "targets":np.zeros((batch_size,length,nt,7),np.float32), "pair":np.zeros((batch_size,length,nb,nt,13),np.float32),
              "red_mask":np.zeros((batch_size,length,nr),bool), "blue_mask":np.zeros((batch_size,length,nb),bool),
              "target_mask":np.zeros((batch_size,length,nt),bool), "context":np.zeros((batch_size,length,2),np.float32),
              "target_labels":np.full((batch_size,length,nb),-100,np.int64),
              "group_labels":np.zeros((batch_size,length,nb,nb),np.float32),
              "lengths":np.asarray(list(map(len,sequences)),np.int64)}
    for i, sequence in enumerate(sequences):
        for t, (state, action) in enumerate(sequence):
            red,blue,targets,pair,context = state_arrays(state)
            for key,value in [("red",red),("blue",blue),("targets",targets)]:
                arrays[key][i,t,:len(value)] = value
                arrays[key+"_mask" if key != "targets" else "target_mask"][i,t,:len(value)] = True
            arrays["pair"][i,t,:len(blue),:len(targets)] = pair
            arrays["context"][i,t] = context
            ids = state.ids("blue")
            targets_by_id = {x.id:j for j,x in enumerate(state.targets)}
            groups = {int(k):(targets_by_id[int(target)],g) for g,(target,members) in enumerate(action["groups"]) for k in members}
            for j,k in enumerate(ids):
                arrays["target_labels"][i,t,j] = groups[k][0]
                for j2,k2 in enumerate(ids):
                    arrays["group_labels"][i,t,j,j2] = float(groups[k][1] == groups[k2][1])
    return {key:torch.as_tensor(value,device=device) for key,value in arrays.items()}


def loss(model, batch):
    targets,groups,vq,index,embedding = model(batch)
    target_loss = F.cross_entropy(targets.reshape(-1,targets.shape[-1]),batch["target_labels"].reshape(-1),ignore_index=-100)
    mask = batch["blue_mask"][..., :,None] & batch["blue_mask"][...,None,:]
    n = groups.shape[-1]
    mask &= ~torch.eye(n,dtype=torch.bool,device=groups.device)
    group_loss = F.binary_cross_entropy_with_logits(groups[mask],batch["group_labels"][mask]) if mask.any() else groups.sum()*0
    total = target_loss+.25*group_loss+vq
    return total,{"target_loss":float(target_loss.detach()),"group_loss":float(group_loss.detach()),"vq_loss":float(vq.detach())},index,embedding


class QOMRuntime:
    def __init__(self, model, device="cpu", prior=None, thresholds=None):
        self.model, self.device = model, torch.device(device)
        self.prior = np.full(model.codes,1/model.codes) if prior is None else np.asarray(prior,float)
        self.thresholds = thresholds or {}
        self.hidden = None

    def copy(self):
        result = QOMRuntime(self.model,self.device,self.prior,self.thresholds)
        result.hidden = None if self.hidden is None else self.hidden.clone()
        return result

    def distributions(self,state):
        red,blue,targets,pair,context = state_arrays(state)
        def tensor(x):
            return torch.as_tensor(x,device=self.device)[None,None]
        with torch.inference_mode():
            h,self.hidden = self.model.encode(tensor(red),tensor(blue),tensor(targets),
                                             tensor(np.ones(len(red),bool)),tensor(np.ones(len(blue),bool)),
                                             tensor(np.ones(len(targets),bool)),tensor(context),self.hidden)
            logits,groups = [],[]
            for code in self.model.codebook.weight:
                a,b = self.model.decode(tensor(blue),tensor(pair),tensor(context),h,code[None],tensor(np.ones(len(targets),bool)))
                logits.append(a.softmax(-1)[0,0].cpu().numpy())
                groups.append(b.sigmoid()[0,0].cpu().numpy())
        return np.asarray(logits),np.asarray(groups)
