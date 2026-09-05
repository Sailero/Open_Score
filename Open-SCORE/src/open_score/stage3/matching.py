"""Map Stage-3 count allocations to concrete agent/task identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class AgentTaskMatching:
    """Minimum-cost assignment with ``-1`` denoting the reserve group."""

    assignment: np.ndarray
    task_counts: Tuple[int, ...]
    task_agents: Tuple[Tuple[Hashable, ...], ...]
    reserve_agents: Tuple[Hashable, ...]
    agent_ids: Tuple[Hashable, ...]
    task_ids: Tuple[Hashable, ...]
    total_cost: float

    def assignment_by_agent(self) -> dict:
        """Return ``agent_id -> task_id``; reserve agents map to ``None``."""

        return {
            agent_id: (None if task < 0 else self.task_ids[int(task)])
            for agent_id, task in zip(self.agent_ids, self.assignment)
        }


def _validate_integer_vector(
    values: Sequence[int], length: int, name: str
) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    result = result.astype(np.int64, copy=True)
    if np.any(result < 0):
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_size(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _prepare_problem(
    n_agents: int,
    n_tasks: int,
    demands: Sequence[int],
    capacities: Optional[Sequence[int]],
    reserve_cost: Optional[Union[float, Sequence[float]]],
    agent_ids: Optional[Sequence[Hashable]],
    task_ids: Optional[Sequence[Hashable]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Tuple[Hashable, ...],
    Tuple[Hashable, ...],
]:
    minimum = _validate_integer_vector(demands, n_tasks, "demands")
    maximum = (
        minimum.copy()
        if capacities is None
        else _validate_integer_vector(capacities, n_tasks, "capacities")
    )
    if np.any(maximum < minimum):
        raise ValueError("capacities must be at least demands")
    if int(minimum.sum()) > n_agents:
        raise ValueError("total task demand exceeds the number of agents")

    if reserve_cost is None:
        reserve = None
    else:
        reserve_array = np.asarray(reserve_cost, dtype=np.float64)
        if reserve_array.ndim == 0:
            reserve = np.full(n_agents, float(reserve_array), dtype=np.float64)
        elif reserve_array.shape == (n_agents,):
            reserve = reserve_array.copy()
        else:
            raise ValueError("reserve_cost must be a scalar or have one value per agent")
        if not np.all(np.isfinite(reserve)):
            raise ValueError("reserve_cost must be finite")
    if reserve is None and int(maximum.sum()) < n_agents:
        raise ValueError("total capacity is too small when reserve is disabled")

    if agent_ids is None:
        stable_agent_ids: Tuple[Hashable, ...] = tuple(range(n_agents))
    else:
        if len(agent_ids) != n_agents:
            raise ValueError("agent_ids must have one entry per cost row")
        stable_agent_ids = tuple(agent_ids)
        if len(set(stable_agent_ids)) != n_agents:
            raise ValueError("agent_ids must be unique")
    if task_ids is None:
        stable_task_ids: Tuple[Hashable, ...] = tuple(range(n_tasks))
    else:
        if len(task_ids) != n_tasks:
            raise ValueError("task_ids must have one entry per cost column")
        stable_task_ids = tuple(task_ids)
        if len(set(stable_task_ids)) != n_tasks:
            raise ValueError("task_ids must be unique")
    return minimum, maximum, reserve, stable_agent_ids, stable_task_ids


def _coalesce_edges(
    edge_agents: np.ndarray,
    edge_tasks: np.ndarray,
    edge_costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the cheapest value for every repeated COO coordinate."""

    if len(edge_agents) < 2:
        return edge_agents, edge_tasks, edge_costs
    # Agent and task are the primary keys; cost is the final tie breaker, so
    # the first row for a repeated coordinate has the minimum finite cost.
    order = np.lexsort((edge_costs, edge_tasks, edge_agents))
    agents = edge_agents[order]
    tasks = edge_tasks[order]
    costs = edge_costs[order]
    first = np.ones(len(agents), dtype=bool)
    first[1:] = (agents[1:] != agents[:-1]) | (tasks[1:] != tasks[:-1])
    return agents[first], tasks[first], costs[first]


def _match_from_edges(
    n_agents: int,
    n_tasks: int,
    edge_agents: np.ndarray,
    edge_tasks: np.ndarray,
    edge_costs: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    reserve: Optional[np.ndarray],
    stable_agent_ids: Tuple[Hashable, ...],
    stable_task_ids: Tuple[Hashable, ...],
) -> AgentTaskMatching:
    """Shared transportation LP for dense and COO public entry points."""

    if reserve is not None:
        edge_agents = np.concatenate(
            (edge_agents, np.arange(n_agents, dtype=np.int64))
        )
        edge_tasks = np.concatenate(
            (edge_tasks, np.full(n_agents, -1, dtype=np.int64))
        )
        edge_costs = np.concatenate((edge_costs, reserve.astype(np.float64, copy=False)))
    n_variables = len(edge_agents)
    if n_variables == 0:
        raise ValueError("the assignment graph has no finite edges")

    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix

        variable_indices = np.arange(n_variables, dtype=np.int64)
        a_eq = coo_matrix(
            (np.ones(n_variables), (edge_agents, variable_indices)),
            shape=(n_agents, n_variables),
        ).tocsr()

        target_variable_indices = np.flatnonzero(edge_tasks >= 0)
        target_columns = edge_tasks[target_variable_indices]
        a_ub = coo_matrix(
            (
                np.r_[
                    np.ones(len(target_variable_indices)),
                    -np.ones(len(target_variable_indices)),
                ],
                (
                    np.r_[target_columns, n_tasks + target_columns],
                    np.r_[target_variable_indices, target_variable_indices],
                ),
            ),
            shape=(2 * n_tasks, n_variables),
        ).tocsr()
        result = linprog(
            edge_costs,
            A_ub=a_ub,
            b_ub=np.r_[maximum, -minimum].astype(np.float64),
            A_eq=a_eq,
            b_eq=np.ones(n_agents, dtype=np.float64),
            bounds=(0.0, 1.0),
            method="highs",
        )
    except ImportError as error:
        raise RuntimeError("SciPy is required for agent-task matching") from error
    if not result.success:
        raise ValueError(f"agent-task matching is infeasible: {result.message}")

    solution = np.asarray(result.x, dtype=np.float64)
    # A transportation-polytope vertex is integral.  Check rather than silently
    # rounding a non-integral result, because rounding could violate a demand.
    if np.max(np.minimum(solution, 1.0 - solution)) > 1e-6:
        raise RuntimeError("assignment solver returned a non-integral optimum")
    assignment = np.full(n_agents, -2, dtype=np.int64)
    selected = np.flatnonzero(solution > 0.5)
    for variable in selected:
        agent = int(edge_agents[variable])
        task = int(edge_tasks[variable])
        if assignment[agent] != -2:
            raise RuntimeError("assignment solver selected two tasks for one agent")
        assignment[agent] = task
    if np.any(assignment == -2):
        raise RuntimeError("assignment solver left an agent unassigned")

    assigned_tasks = assignment[assignment >= 0]
    count_array = np.bincount(assigned_tasks, minlength=n_tasks)
    if np.any(count_array < minimum) or np.any(count_array > maximum):
        raise RuntimeError("assignment reconstruction violated task bounds")
    task_agent_lists = [[] for _ in range(n_tasks)]
    reserve_agents_list = []
    for index, task in enumerate(assignment):
        if task < 0:
            reserve_agents_list.append(stable_agent_ids[index])
        else:
            task_agent_lists[int(task)].append(stable_agent_ids[index])
    selected_cost = float(edge_costs[selected].sum())
    assignment.setflags(write=False)
    return AgentTaskMatching(
        assignment=assignment,
        task_counts=tuple(int(value) for value in count_array),
        task_agents=tuple(tuple(values) for values in task_agent_lists),
        reserve_agents=tuple(reserve_agents_list),
        agent_ids=stable_agent_ids,
        task_ids=stable_task_ids,
        total_cost=selected_cost,
    )


def match_agents_to_tasks(
    cost_matrix: np.ndarray,
    demands: Sequence[int],
    *,
    capacities: Optional[Sequence[int]] = None,
    reserve_cost: Optional[Union[float, Sequence[float]]] = None,
    agent_ids: Optional[Sequence[Hashable]] = None,
    task_ids: Optional[Sequence[Hashable]] = None,
) -> AgentTaskMatching:
    """Solve a sparse minimum-cost assignment with lower/upper task counts.

    Args:
        cost_matrix: ``[N, M]`` costs.  ``+inf`` removes an agent-task edge,
            which lets callers retain only a few ETA-nearest candidate tasks.
        demands: Required minimum number of agents per task.
        capacities: Maximum number per task.  Defaults to ``demands``, which
            realizes a Stage-3 count allocation exactly.
        reserve_cost: Scalar or length-``N`` cost of leaving an agent in the
            reserve.  ``None`` disables reserve edges.
        agent_ids: Optional stable environment IDs; defaults to row indices.
        task_ids: Optional target/threat-cell IDs; defaults to column indices.

    The problem is a transportation LP.  Its constraint matrix is totally
    unimodular, so a HiGHS basic optimum is integral without a combinatorial
    agent-permutation search.  Sparse ``inf`` edges keep 1000-agent matching
    practical.
    """

    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] <= 0 or costs.shape[1] <= 0:
        raise ValueError("cost_matrix must have non-empty shape [N, M]")
    if np.any(np.isnan(costs)) or np.any(np.isneginf(costs)):
        raise ValueError("cost_matrix may contain finite values or +inf only")
    n_agents, n_tasks = costs.shape
    finite_agents, finite_tasks = np.nonzero(np.isfinite(costs))
    return match_sparse_agents_to_tasks(
        n_agents,
        n_tasks,
        finite_agents,
        finite_tasks,
        costs[finite_agents, finite_tasks],
        demands,
        capacities=capacities,
        reserve_cost=reserve_cost,
        agent_ids=agent_ids,
        task_ids=task_ids,
    )


def match_sparse_agents_to_tasks(
    n_agents: int,
    n_tasks: int,
    edge_agent_indices: Sequence[int],
    edge_task_indices: Sequence[int],
    edge_costs: Sequence[float],
    demands: Sequence[int],
    *,
    capacities: Optional[Sequence[int]] = None,
    reserve_cost: Optional[Union[float, Sequence[float]]] = None,
    agent_ids: Optional[Sequence[Hashable]] = None,
    task_ids: Optional[Sequence[Hashable]] = None,
) -> AgentTaskMatching:
    """Solve matching directly from a COO agent--task edge list.

    Unlike :func:`match_agents_to_tasks`, this entry point never constructs an
    ``N x M`` cost array.  The three edge sequences describe finite entries as
    parallel COO coordinates and values.  Repeated coordinates are accepted
    and coalesced to their minimum cost.  Reserve edges remain implicit and
    are created once per agent only when ``reserve_cost`` is supplied.

    Example::

        result = match_sparse_agents_to_tasks(
            3, 2,
            edge_agent_indices=[0, 1, 2],
            edge_task_indices=[0, 1, 0],
            edge_costs=[0.2, 0.1, 0.4],
            demands=[1, 1],
            reserve_cost=0.0,
        )
    """

    agents = _positive_size(n_agents, "n_agents")
    tasks = _positive_size(n_tasks, "n_tasks")

    edge_agents_raw = np.asarray(edge_agent_indices)
    edge_tasks_raw = np.asarray(edge_task_indices)
    costs = np.asarray(edge_costs, dtype=np.float64)
    if edge_agents_raw.ndim != 1 or edge_tasks_raw.ndim != 1 or costs.ndim != 1:
        raise ValueError("COO edge agents, tasks and costs must be one-dimensional")
    if not (len(edge_agents_raw) == len(edge_tasks_raw) == len(costs)):
        raise ValueError("COO edge agents, tasks and costs must have equal length")
    if len(edge_agents_raw):
        if not np.issubdtype(edge_agents_raw.dtype, np.integer):
            raise TypeError("edge_agent_indices must contain integers")
        if not np.issubdtype(edge_tasks_raw.dtype, np.integer):
            raise TypeError("edge_task_indices must contain integers")
    edge_agents = edge_agents_raw.astype(np.int64, copy=True)
    edge_tasks = edge_tasks_raw.astype(np.int64, copy=True)
    if np.any((edge_agents < 0) | (edge_agents >= agents)):
        raise ValueError("edge_agent_indices contain an out-of-range index")
    if np.any((edge_tasks < 0) | (edge_tasks >= tasks)):
        raise ValueError("edge_task_indices contain an out-of-range index")
    if not np.all(np.isfinite(costs)):
        raise ValueError("COO edge costs must be finite")
    edge_agents, edge_tasks, costs = _coalesce_edges(
        edge_agents, edge_tasks, costs.astype(np.float64, copy=True)
    )
    minimum, maximum, reserve, stable_agent_ids, stable_task_ids = _prepare_problem(
        agents,
        tasks,
        demands,
        capacities,
        reserve_cost,
        agent_ids,
        task_ids,
    )
    return _match_from_edges(
        agents,
        tasks,
        edge_agents,
        edge_tasks,
        costs,
        minimum,
        maximum,
        reserve,
        stable_agent_ids,
        stable_task_ids,
    )


__all__ = [
    "AgentTaskMatching",
    "match_agents_to_tasks",
    "match_sparse_agents_to_tasks",
]
