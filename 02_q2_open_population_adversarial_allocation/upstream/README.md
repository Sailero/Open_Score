# Upstream sources and reproduction status

## Primary hierarchical base: ALMA

- Paper: https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html
- Code: https://github.com/shariqiqbal2810/ALMA
- Use: high-level allocation, subtask masking, REFIL/QMIX lower policies, SaveTheCity and composite StarCraft, COPA and heuristic baselines.
- Audit: old Python 3.7/PyMARL/SC2 stack; reproduce one author configuration within three days before adopting it as the main codebase.

## Current openness baseline: PLATO

- Paper: https://arxiv.org/abs/2607.25082
- Use: joint agent/task openness definition, pointer actor and variable graph critic, zero-shot openness evaluation.
- Audit: no author repository was located during the 2026-08-27 search; implement only a documented minimal baseline.

## Nearest engineering baseline: DT-GAT-MARL

- Paper: https://doi.org/10.1038/s41598-026-55576-9
- Use: dynamic-topology allocation, MAPPO executor, dual-frequency hierarchy, effective-reallocation and oscillation metrics.
- Audit: publication states that code/data are available on reasonable request; no clone-ready repository is assumed.

## Modern fallback

- JaxMARL: https://github.com/bold-lab-ai/JaxMARL
- MAgent2: https://github.com/Farama-Foundation/MAgent2
- MOHITO/FreeRangeZoo openness reference: https://github.com/oasys-mas/mohito-public

JaxMARL is the preferred engineering fallback if ALMA fails the reproduction gate. MAgent2 is a scale stress test, not the primary hierarchical environment. Upstream repositories should be cloned on the Linux/WSL training environment with exact commit hashes; do not copy unlicensed source into this directory.
