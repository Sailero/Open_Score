"""Fixed potential shaping with zero terminal potential and native win reporting."""
import numpy as np


def potential(state, terminal=False):
    # True terminals include the native 50-step horizon. A training/rollout
    # boundary is NOT terminal. With gamma=1 this telescopes to -Phi(initial),
    # preserving the ordering of policies by terminal success probability.
    if terminal:
        return 0.0
    red, blue, targets = state.alive('red'), state.alive('blue'), state.alive('targets')
    total_blue = max(1, len(state.blue))
    blue_survival = len(blue) / total_blue
    if not blue or not targets:
        return -blue_survival
    # Continuous proximity matters because target HP is almost binary under
    # the native damage model. Only current public positions/health are used.
    danger = np.mean([np.exp(-min(np.linalg.norm(np.asarray(b.position)-t.position)
                                  for t in targets)/1000.0) for b in blue])
    interception = (np.mean([np.exp(-min(np.linalg.norm(np.asarray(b.position)-r.position)
                                         for r in red)/800.0) for b in blue]) if red else 0.0)
    return float(-0.5*blue_survival - 0.25*danger + 0.15*interception)


def shaped_reward(state, following, native_reward, done):
    return float(native_reward) + potential(following, terminal=done) - potential(state)
