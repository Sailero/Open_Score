"""Compatibility loaders for retained v2 policy checkpoints."""
from __future__ import annotations

import torch
from .policy import GroupingPolicy


def make_policy(config):
    method = config.get('method', 'selective')
    model = dict(config.get('model', {}))
    if method == 'alma':
        from .baselines import AlmaStylePolicy
        return AlmaStylePolicy(**model, seed=config.get('seed', 20260905)).to(config.get('device', 'cpu'))
    if method in ('static', 'dlom'):
        from .baselines import StaticPolicy, DLOMSearchPolicy
        return StaticPolicy() if method == 'static' else DLOMSearchPolicy(device=config.get('device', 'cpu'))
    return GroupingPolicy(mode=method, **model).to(config.get('device', 'cpu'))


def load_policy(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = dict(payload['config'])
    config['device'] = device
    policy = make_policy(config)
    policy.load_state_dict(payload['model'])
    if config['method'] == 'random' and payload.get('release_distribution'):
        policy.set_release_distribution(payload['release_distribution'])
    policy.eval()
    return policy, payload
