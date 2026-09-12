"""Masked entity adaptation of official SPECTra GRF ST-HyperNet mixer.
Source: funny-rl/SPECTra ffababf6187216c9d16b2109ee8ef6fe5fdf1172.
Retains query-key generators, learned pooling seed, two hypernetworks,
ELU and absolute weights. Removes football goalkeeper/single-ball slicing.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .spectra_attention import QueryKeyBlock, CrossAttentionBlock, PoolingQueryKeyBlock


class ST_HyperNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        width = args.hypernet_embed
        dim = args.entity_shape + (args.n_actions if args.entity_last_action else 0)
        self.agent_embedding = nn.Linear(dim, width)
        self.enemy_embedding = nn.Linear(dim, width)
        self.target_embedding = nn.Linear(dim, width)
        self.cross_attention = CrossAttentionBlock(d=width, h=args.mixing_n_head)
        self.weight_mlp = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        self.bias_mlp = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        self.weight_generator = QueryKeyBlock(d=width, h=args.mixing_n_head)
        self.bias_generator = PoolingQueryKeyBlock(d=width, k=1, h=args.mixing_n_head)

    def forward(self, inputs):
        state = inputs["entities"]
        mask = inputs["entity_mask"].bool()
        state = state.reshape(-1, state.shape[-2], state.shape[-1])
        mask = mask.reshape(-1, mask.shape[-1])
        valid = ~mask
        if getattr(self.args, "feature_layout", "had") == "had":
            embed = (self.agent_embedding(state) * state[..., 7:8]
                     + self.enemy_embedding(state) * state[..., 8:9]
                     + self.target_embedding(state) * state[..., 9:10])
        else:
            embed = self.agent_embedding(state)
        embed = embed.masked_fill(mask[..., None], 0)
        a_embed = embed[:, :self.n_agents]
        agent_valid = valid[:, :self.n_agents]
        visible = valid[:, None].expand(-1, self.n_agents, -1)
        x = self.cross_attention(a_embed, embed, visible)
        x = x.masked_fill(~agent_valid[..., None], 0)
        weight_x = (x + self.weight_mlp(x)).masked_fill(~agent_valid[..., None], 0)
        bias_x = (x + self.bias_mlp(x)).masked_fill(~agent_valid[..., None], 0)
        weight = self.weight_generator(weight_x, weight_x)
        weight = weight.masked_fill(~(agent_valid[:, :, None] & agent_valid[:, None, :]), 0)
        bias = self.bias_generator(bias_x).masked_fill(~agent_valid[:, None, :], 0)
        return weight, bias


class SPECTraMixer(nn.Module):
    def __init__(self, args, abs=True):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hyper_w1 = ST_HyperNet(args)
        self.hyper_w2 = ST_HyperNet(args)
        self.abs = abs

    def forward(self, qvals, states, imagine_groups=None):
        b, t, _ = qvals.shape
        active = ~states["entity_mask"][..., :self.n_agents].bool()
        qvals = qvals.masked_fill(~active, 0).reshape(b * t, 1, self.n_agents)
        w1, b1 = self.hyper_w1(states)
        w2, b2 = self.hyper_w2(states)
        if self.abs:
            w1, w2 = self.pos_func(w1), self.pos_func(w2)
        h1 = F.elu(th.matmul(qvals, w1) + b1)
        h2 = (th.matmul(h1, w2) + b2).sum(dim=-1, keepdim=False)
        return h2.view(b, t, 1)

    def pos_func(self, x):
        return th.abs(x)

    def denormalize(self, q):
        return q

