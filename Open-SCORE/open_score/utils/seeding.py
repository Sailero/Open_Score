"""Independent model, episode, scale, and held-out evaluation RNG streams."""
from __future__ import annotations

import random
import numpy as np

EVAL_SEEDS = tuple(range(9500, 9600))
FINAL_EVAL_SEEDS = tuple(range(9000, 9300))


def split_seeds(seed, worker_id=0):
    streams = np.random.SeedSequence([int(seed), int(worker_id)]).spawn(3)
    values = [int(stream.generate_state(1)[0]) for stream in streams]
    return {"train_seed": int(seed), "environment_seed": values[0],
            "scale_seed": values[1], "evaluation_seed": values[2]}


def seed_everything(seed, deterministic=False):
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True)


class EpisodeSeedStream:
    def __init__(self, seed):
        self.rng = np.random.default_rng(int(seed))

    def next(self):
        # Keep all training initial conditions outside frozen validation/anchor IDs.
        return int(self.rng.integers(100_000, 2 ** 31 - 1))

    def state_dict(self):
        return self.rng.bit_generator.state

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state
