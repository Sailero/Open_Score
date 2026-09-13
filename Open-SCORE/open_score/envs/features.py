"""Scale-independent HAD features; episode buffers store one absolute table."""
from __future__ import annotations

import numpy as np

MAX_AGENTS = 40
MAX_BLUE = 40
MAX_TARGETS = 6
MAX_ENTITIES = MAX_AGENTS + MAX_BLUE + MAX_TARGETS
TRAIN_AGENTS = 10
TRAIN_BLUE = 10
TRAIN_TARGETS = 3
TRAIN_ENTITIES = TRAIN_AGENTS + TRAIN_BLUE + TRAIN_TARGETS
N_ACTIONS = 9
ENTITY_DIM = 10
POSITION_SCALE = 2500.0
VELOCITY_SCALE = 120.0
FEATURE_NAMES = ("x", "y", "vx", "vy", "alive", "health", "target_damage_share",
                 "is_red", "is_blue", "is_target")


def resolve_pad(pad):
    """Train uses the pool ceiling; eval keeps the 40-red / 40-blue interface."""
    if pad in (None, "eval"):
        return MAX_AGENTS, MAX_BLUE, MAX_TARGETS
    if pad == "train":
        return TRAIN_AGENTS, TRAIN_BLUE, TRAIN_TARGETS
    if isinstance(pad, (tuple, list)) and len(pad) == 3:
        counts = tuple(int(value) for value in pad)
        limits = (MAX_AGENTS, MAX_BLUE, MAX_TARGETS)
        if any(not 0 < count <= limit for count, limit in zip(counts, limits)):
            raise ValueError(f"slot budget {counts} is outside the entity interface {limits}")
        return counts
    raise ValueError(f"unknown entity pad {pad!r}")


def slot_indices(n_red=MAX_AGENTS, n_blue=MAX_BLUE, n_targets=MAX_TARGETS,
                 table_red=None, table_blue=None):
    """Rows of a packed red/blue/target table kept by a reduced slot budget."""
    table_red = MAX_AGENTS if table_red is None else int(table_red)
    table_blue = MAX_BLUE if table_blue is None else int(table_blue)
    counts = (int(n_red), int(n_blue), int(n_targets))
    limits = (table_red, table_blue, MAX_TARGETS)
    if any(not 0 < count <= limit for count, limit in zip(counts, limits)):
        raise ValueError(f"slot budget {counts} is outside the entity table {limits}")
    starts = (0, table_red, table_red + table_blue)
    return np.concatenate([np.arange(start, start + count)
                           for start, count in zip(starts, counts)])


def pool_slot_indices(args):
    """Kept rows for an ordered baseline whose slots match the training pool.

    Returns ``None`` when the live table is already the pool, or when the
    method uses the whole entity interface.
    """
    budget = getattr(args, "pool_slots", None)
    if budget is None or getattr(args, "feature_layout", "had") != "had":
        return None
    n_red = int(getattr(args, "n_agents", MAX_AGENTS))
    n_tasks = int(getattr(args, "n_tasks", MAX_TARGETS))
    n_entities = int(getattr(args, "n_entities", MAX_ENTITIES))
    n_blue = n_entities - n_red - n_tasks
    if tuple(int(value) for value in budget) == (n_red, n_blue, n_tasks):
        return None
    return slot_indices(*budget, table_red=n_red, table_blue=n_blue)


def planar_action_mapping(primitives):
    """Derive both numbering systems from HAD's immutable official ordering."""
    values = np.asarray(primitives)
    native_ids = np.flatnonzero(values[:, 2] == 0).astype(np.int64)
    if values.shape != (27, 3) or len(native_ids) != N_ACTIONS or native_ids[0] != 0:
        raise ValueError("HAD planar action contract has changed")
    inverse = np.full(27, -1, dtype=np.int64)
    inverse[native_ids] = np.arange(N_ACTIONS)
    if not np.array_equal(inverse[native_ids], np.arange(N_ACTIONS)):
        raise AssertionError("planar action mapping is not bijective")
    return native_ids, inverse


def entities_from_state(state):
    """Convert physical state to fixed red/blue/target slots, preserving IDs."""
    entities = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
    entity_mask = np.ones(MAX_ENTITIES, dtype=np.uint8)
    for side, start, limit, kind in (("red", 0, MAX_AGENTS, 0),
                                     ("blue", MAX_AGENTS, MAX_BLUE, 1),
                                     ("targets", MAX_AGENTS + MAX_BLUE, MAX_TARGETS, 2)):
        rows = tuple(getattr(state, side))
        if len(rows) > limit:
            raise ValueError(f"{side} count exceeds experiment interface ({limit})")
        live = np.asarray([row.alive or kind == 2 for row in rows], dtype=bool)
        if not len(rows):
            continue
        indices = np.arange(start, start + len(rows))[live]
        rows = [row for row, alive in zip(rows, live) if alive]
        if not rows:
            continue
        entities[indices, :2] = np.asarray([row.position[:2] for row in rows]) / POSITION_SCALE
        entities[indices, 7 + kind] = 1.0
        entity_mask[indices] = 0
        if kind == 2:
            entities[indices, 6] = [row.cumulative_damage / len(state.blue) for row in rows]
        else:
            entities[indices, 2:4] = np.asarray([row.velocity[:2] for row in rows]) / VELOCITY_SCALE
            entities[indices, 4] = 1.0
            entities[indices, 5] = [row.health / row.initial_health for row in rows]
    return entities, entity_mask


def masks_from_entity_mask(entity_mask):
    absent = np.asarray(entity_mask, dtype=np.uint8)
    return {"obs_mask": np.maximum(absent[:, None], absent[None, :]),
            "entity_mask": absent.copy()}


def blue_nearest_from_table(entities, entity_mask, n_red=MAX_AGENTS,
                            n_blue=MAX_BLUE, n_targets=MAX_TARGETS):
    """Public nearest-target index of each Blue slot; -1 if that slot is empty."""
    absent = np.asarray(entity_mask, dtype=bool)
    nearest = np.full(int(n_blue), -1, dtype=np.int32)
    live_blue = np.flatnonzero(~absent[n_red:n_red + n_blue])
    live_tgt = np.flatnonzero(~absent[n_red + n_blue:n_red + n_blue + n_targets])
    if not len(live_blue) or not len(live_tgt):
        return nearest
    table = np.asarray(entities)
    distances = np.linalg.norm(table[n_red + live_blue, None, :2]
                               - table[n_red + n_blue + live_tgt, :2][None], axis=-1)
    nearest[live_blue] = live_tgt[distances.argmin(axis=1)]
    return nearest


def task_masks(entities, entity_mask, n_active_targets, n_red=MAX_AGENTS,
               n_blue=MAX_BLUE, n_targets=MAX_TARGETS, subtask_set="targets"):
    """ALMA's subtask contract.

    ``targets``: one subtask per live target. Blue rows attach to the target
    they are closest to, a quantity every method can derive from the same
    entity table; the Blue side's own assignment is private and is never used.

    ``blues``: one subtask per Blue slot. A live Blue belongs only to its own
    slot; live targets are shared context for every currently active Blue
    task. Empty or dead Blue slots are inactive tasks.

    Red rows start attached to every active subtask and are replaced by the
    allocation at decision points. Zero means "belongs to".
    """
    if subtask_set not in ("targets", "blues"):
        raise ValueError(f"unknown subtask set {subtask_set!r}")
    absent = np.asarray(entity_mask, dtype=np.uint8).astype(bool)
    if subtask_set == "blues":
        task_mask = absent[n_red:n_red + n_blue].astype(np.uint8).copy()
        entity2task = np.ones((n_red + n_blue + n_targets, n_blue), dtype=np.uint8)
        active = np.flatnonzero(task_mask == 0)
        if len(active):
            entity2task[np.ix_(np.flatnonzero(~absent[:n_red]), active)] = 0
            live_targets = np.flatnonzero(~absent[n_red + n_blue:n_red + n_blue + n_targets]) + n_red + n_blue
            if len(live_targets):
                entity2task[np.ix_(live_targets, active)] = 0
            entity2task[n_red + active, active] = 0
        return {"task_mask": task_mask, "entity2task_mask": entity2task}
    task_mask = np.ones(n_targets, dtype=np.uint8)
    task_mask[:int(n_active_targets)] = 0
    entity2task = np.ones((n_red + n_blue + n_targets, n_targets), dtype=np.uint8)
    active = np.arange(int(n_active_targets))
    if not len(active):
        raise ValueError("a HAD configuration always defends at least one target")
    entity2task[np.ix_(np.flatnonzero(~absent[:n_red]), active)] = 0
    target_rows = n_red + n_blue + active
    entity2task[target_rows, active] = 0
    live_blue = np.flatnonzero(~absent[n_red:n_red + n_blue]) + n_red
    if len(live_blue):
        distances = np.linalg.norm(np.asarray(entities)[live_blue, None, :2]
                                   - np.asarray(entities)[target_rows, :2][None], axis=-1)
        entity2task[live_blue, active[distances.argmin(axis=1)]] = 0
    return {"task_mask": task_mask, "entity2task_mask": entity2task}


def available_actions(entity_mask, n_agents=None):
    n_agents = MAX_AGENTS if n_agents is None else int(n_agents)
    avail = np.repeat((1 - np.asarray(entity_mask[:n_agents]))[:, None], N_ACTIONS, axis=1)
    avail[:, 0] = 1
    return avail.astype(np.int64)
