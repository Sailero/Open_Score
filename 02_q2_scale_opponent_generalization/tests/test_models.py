from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from q2marl import HPNPolicy, OpponentConditionedHPN


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch = 3
        self.self_features = torch.randn(self.batch, 8)
        self.entities = torch.randn(self.batch, 7, 16)
        self.entity_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]],
            dtype=torch.bool,
        )
        self.enemies = torch.randn(self.batch, 5, 16)
        self.enemy_mask = torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool
        )

    def _run(self, model):
        model.eval()
        return model(
            self.self_features,
            self.entities,
            self.entity_mask,
            self.enemies,
            self.enemy_mask,
        )

    def test_entity_permutation_invariance(self):
        model = HPNPolicy()
        out = self._run(model)
        permutation = torch.tensor([3, 0, 6, 2, 1, 5, 4])
        permuted = model(
            self.self_features,
            self.entities[:, permutation],
            self.entity_mask[:, permutation],
            self.enemies,
            self.enemy_mask,
        )
        torch.testing.assert_close(out.fixed_logits, permuted.fixed_logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out.value, permuted.value, atol=1e-5, rtol=1e-5)

    def test_enemy_permutation_equivariance(self):
        model = HPNPolicy()
        out = self._run(model)
        permutation = torch.tensor([2, 4, 0, 3, 1])
        permuted = model(
            self.self_features,
            self.entities,
            self.entity_mask,
            self.enemies[:, permutation],
            self.enemy_mask[:, permutation],
        )
        torch.testing.assert_close(
            out.target_logits[:, permutation], permuted.target_logits, atol=1e-5, rtol=1e-5
        )

    def test_padding_is_ignored(self):
        model = HPNPolicy()
        model.eval()
        base = model(
            self.self_features[:1],
            self.entities[:1, :4],
            self.entity_mask[:1, :4],
            self.enemies[:1, :3],
            self.enemy_mask[:1, :3],
        )
        padded_entities = torch.cat([self.entities[:1, :4], torch.randn(1, 6, 16)], dim=1)
        padded_entity_mask = torch.cat(
            [self.entity_mask[:1, :4], torch.zeros(1, 6, dtype=torch.bool)], dim=1
        )
        padded_enemies = torch.cat([self.enemies[:1, :3], torch.randn(1, 4, 16)], dim=1)
        padded_enemy_mask = torch.cat(
            [self.enemy_mask[:1, :3], torch.zeros(1, 4, dtype=torch.bool)], dim=1
        )
        padded = model(
            self.self_features[:1],
            padded_entities,
            padded_entity_mask,
            padded_enemies,
            padded_enemy_mask,
        )
        torch.testing.assert_close(base.fixed_logits, padded.fixed_logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(base.value, padded.value, atol=1e-5, rtol=1e-5)

    def test_oc_hpn_context_and_gradients(self):
        model = OpponentConditionedHPN()
        out = self._run(model)
        self.assertEqual(tuple(out.opponent_state.shape), (self.batch, 32))
        self.assertEqual(tuple(out.auxiliary_prediction.shape), (self.batch, 32))
        loss = out.fixed_logits.mean() + out.value.mean() + out.auxiliary_prediction.square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.opponent_context.parameters()))

    def test_parameter_count_is_size_independent(self):
        model = OpponentConditionedHPN()
        count_before = sum(p.numel() for p in model.parameters())
        _ = self._run(model)
        _ = model(
            torch.randn(2, 8),
            torch.randn(2, 23, 16),
            torch.ones(2, 23, dtype=torch.bool),
            torch.randn(2, 19, 16),
            torch.ones(2, 19, dtype=torch.bool),
        )
        self.assertEqual(count_before, sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()

