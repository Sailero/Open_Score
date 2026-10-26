"""Stable legacy grouping protocol IDs and scheduling-independent seeds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

VERSION = "known-rule-mixed-v5.1"


def stable_seed(*parts):
    raw = json.dumps(parts, sort_keys=True, separators=(',', ':'), default=str).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'little') % (2**31-1)


@dataclass(frozen=True)
class EpisodeSpec:
    family_id: str
    red_count: int
    blue_count: int
    opening_seed: int
    opponent_seed: int
    split: str
    opponent: str = 'reactive'
    protocol_version: str = VERSION

    def __post_init__(self):
        if self.red_count < 1 or self.blue_count < 1:
            raise ValueError('Initial Red and Blue counts must be positive')
        if self.opponent != 'reactive' or self.protocol_version != VERSION:
            raise ValueError('Unsupported frozen opponent/protocol')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
