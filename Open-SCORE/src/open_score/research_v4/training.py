"""Compatibility loader for retained v4 upper-policy checkpoints."""
from __future__ import annotations

from pathlib import Path
import torch
from .learning import CandidateNetwork


class LearnedPolicy:
    def __init__(self, network, config):
        self.network, self.config = network.eval(), config

    @torch.no_grad()
    def act(self, state):
        from .actions import candidate_pool
        pools = candidate_pool(state, budget=self.config['candidate_budget'])
        scores, _ = self.network([state], [pools])
        return pools[int(scores[0].argmax())]

    __call__ = act


def load_policy(checkpoint, device='cpu'):
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if payload.get('schema') != 'research-v4-upper-1':
        raise ValueError('Not a v4 upper checkpoint')
    config = dict(payload['config'], device=str(device))
    actor = CandidateNetwork(**config['model']).to(device)
    actor.load_state_dict(payload['actor'])
    return LearnedPolicy(actor, config)
