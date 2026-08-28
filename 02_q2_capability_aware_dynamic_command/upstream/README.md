# Upstream paper/code map

The local `open_score` package is a small Windows-friendly implementation of the
paper interfaces.  Official repositories remain the source of truth for
baseline reproduction and attribution.

| Stage | Paper/source | Repository | Planned use |
|---|---|---|---|
| S1 | QMIX / PyMARL | https://github.com/oxwhirl/pymarl | Fixed-Capacity QMIX, VDN, IQL semantics and official anchor |
| S1 | REFIL | https://github.com/shariqiqbal2810/REFIL | Attention-QMIX/REFIL variable-entity baseline |
| S2–S4 | ALMA | https://github.com/shariqiqbal2810/ALMA | AQL allocator, heuristic and joint-training baseline |
| S2 | Set Transformer | https://github.com/juho-lee/set_transformer | Bilateral masked attention encoder reference |
| S2 | DFL learning-to-rank | https://github.com/JayMan91/ltr-predopt | Pairwise/listwise decision-focused baseline |
| S4 | HARL | https://github.com/PKU-MARL/HARL | Sequential-update baseline/implementation reference |

Before formal experiments, record for every repository: commit hash, license,
Python/PyTorch/SC2 versions, local patch, command, seed and reproduced metric.
Do not report an adapted HAD result as an official upstream reproduction.

Recommended local layout (ignored by Git):

```text
upstream/external/pymarl/
upstream/external/refil/
upstream/external/alma/
upstream/external/harl/
```

Native Windows is used for Open-HAD and the local PyTorch learner.  The legacy
ALMA/REFIL Docker/StarCraft stacks and later JAX/SMAX work should be isolated in
WSL2.  This prevents incompatible Python and CUDA dependencies from changing
the HAD development environment.
