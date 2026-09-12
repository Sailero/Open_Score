"""Official DCG feature RNN with its flat input map replaced by REFIL attention.
Source: wendelinboehmer/dcg 4de100cddf7c3a7035cd89a47d7c1b8a878e7428.
"""
import torch as th
import torch.nn as nn
from open_score.models.entity_encoder import AttentionEncoder


class RNNFeatureAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.native = getattr(args, "feature_layout", "had") == "native"
        if self.native:
            # rel_overgen stores one original local observation per agent.
            # Preserve the official native input projection for E0.
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        else:
            self.encoder = AttentionEncoder(input_shape, args)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        return next(self.parameters()).new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        if self.native:
            x = nn.functional.relu(self.fc1(inputs["entities"][:, 0, :self.args.n_agents]))
        else:
            x = self.encoder(inputs)[:, 0]
        h = self.rnn(x.reshape(-1, self.args.rnn_hidden_dim),
                     hidden_state.reshape(-1, self.args.rnn_hidden_dim))
        # Persistent slot identity is preserved; death does not remove a node.
        dead = inputs["entity_mask"][:, 0, :self.args.n_agents].bool()
        h = h.reshape(x.shape[0], self.args.n_agents, -1).masked_fill(dead[..., None], 0)
        return None, h

