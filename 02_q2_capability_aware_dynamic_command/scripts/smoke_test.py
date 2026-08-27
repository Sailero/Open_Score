from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from q2marl import HPNPolicy, OpponentConditionedHPN


def main() -> None:
    torch.manual_seed(0)
    batch, n_entities, n_enemies = 4, 12, 7
    inputs = dict(
        self_features=torch.randn(batch, 8),
        entities=torch.randn(batch, n_entities, 16),
        entity_mask=torch.ones(batch, n_entities, dtype=torch.bool),
        enemies=torch.randn(batch, n_enemies, 16),
        enemy_mask=torch.ones(batch, n_enemies, dtype=torch.bool),
    )
    for name, model in [("HPN", HPNPolicy()), ("OC-HPN", OpponentConditionedHPN())]:
        output = model(**inputs)
        parameters = sum(p.numel() for p in model.parameters())
        print(
            f"{name}: parameters={parameters:,}, fixed={tuple(output.fixed_logits.shape)}, "
            f"targets={tuple(output.target_logits.shape)}, value={tuple(output.value.shape)}"
        )


if __name__ == "__main__":
    main()
