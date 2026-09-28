"""Native spatial controls and their order-preserving planar subset."""
from itertools import product

import numpy as np


def _make_acceleration_primitives() -> np.ndarray:
    """Return zero plus the 26 normalised directions in {-1, 0, 1}^3.

    These are quantised *acceleration controls*. They contain no pursuit,
    target-selection, or other hand-written tactical semantics. HAD multiplies
    them by each agent's acceleration limit before advancing the dynamics.
    """

    vectors = [np.zeros(3, dtype=np.float32)]
    for components in product((-1.0, 0.0, 1.0), repeat=3):
        value = np.asarray(components, dtype=np.float32)
        norm = float(np.linalg.norm(value))
        if norm > 0.0:
            vectors.append(value / norm)
    return np.stack(vectors)


ACCELERATION_PRIMITIVES = _make_acceleration_primitives()
PLANAR_ACCELERATION_PRIMITIVES = ACCELERATION_PRIMITIVES[
    ACCELERATION_PRIMITIVES[:, 2] == 0
].copy()
