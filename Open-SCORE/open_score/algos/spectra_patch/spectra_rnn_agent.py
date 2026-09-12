"""HAD adaptation of SPECTra's official GRF recurrent agent.
Source: funny-rl/SPECTra ffababf6187216c9d16b2109ee8ef6fe5fdf1172.
Retained: typed projections, SAQA cross attention, GRU, and action MLP.
Adapted: explicit padding, independent target type, one 9-action output
replacing football-only passing actions and their hard-coded insertion.
"""
import torch as th
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .spectra_attention import CrossAttentionBlock
from open_score.models.entity_encoder import relative_entity_views


class SPECTra_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.hidden_size = args.hidden_size
        self.own_embedding = nn.Linear(input_shape, self.hidden_size)
        self.allies_embedding = nn.Linear(input_shape, self.hidden_size)
        self.enemies_embedding = nn.Linear(input_shape, self.hidden_size)
        self.targets_embedding = nn.Linear(input_shape, self.hidden_size)
        self.entity_attention = CrossAttentionBlock(d=self.hidden_size, h=args.n_head)
        self.rnn = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.normal_actions_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_actions))

    def init_hidden(self):
        return self.own_embedding.weight.new_zeros(1, self.hidden_size)

    def _encode(self, views, hidden, dead):
        if getattr(self.args, "feature_layout", "had") == "had":
            embed = (self.allies_embedding(views) * views[..., 7:8]
                     + self.enemies_embedding(views) * views[..., 8:9]
                     + self.targets_embedding(views) * views[..., 9:10])
        else:
            # Native FF carries its own type one-hot, with no Blue semantics.
            embed = self.allies_embedding(views)
        own = self.own_embedding(views[:, :1])
        embed = th.cat((own, embed[:, 1:]), dim=1)
        valid = (~hidden).clone()
        valid[dead, 0] = True
        out = self.entity_attention(own, embed, valid[:, None])[:, 0]
        return out.masked_fill(dead[:, None], 0)

    def forward(self, inputs, hidden_state=None):
        views, hidden, dead, shape = relative_entity_views(inputs, self.args)
        encoded = []
        for start in range(0, len(views), 256):
            x, m, d = views[start:start + 256], hidden[start:start + 256], dead[start:start + 256]
            encoded.append(checkpoint(self._encode, x, m, d, use_reentrant=False)
                           if self.training and th.is_grad_enabled() else self._encode(x, m, d))
        encoded = th.cat(encoded).reshape(*shape, self.hidden_size)
        bs, ts, na = shape
        if hidden_state is None:
            hidden_state = encoded.new_zeros(bs, na, self.hidden_size)
        h = hidden_state.reshape(bs * na, self.hidden_size)
        qs = []
        for t in range(ts):
            h = self.rnn(encoded[:, t].reshape(bs * na, self.hidden_size), h)
            inactive = inputs["entity_mask"][:, t, :na].bool().reshape(-1, 1)
            h = h.masked_fill(inactive, 0)
            q = self.normal_actions_net(h).masked_fill(inactive, 0)
            qs.append(q.reshape(bs, na, self.n_actions))
            # Keep final-observation memory for time-limit bootstrapping;
            # episodes are not concatenated in this protocol's replay.
        return th.stack(qs, dim=1), h.reshape(bs, na, self.hidden_size)

