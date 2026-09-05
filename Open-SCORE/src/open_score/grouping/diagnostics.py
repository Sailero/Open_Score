"""Small physical E1/E2/E3 probes; no diagnostic rollout is free online data."""
from __future__ import annotations

import copy
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
import torch

from .baselines import DLOMSearchPolicy, StaticPolicy, balanced_initial, candidates
from .domain import Group, Grouping
from .environment import KnownOpponentEnv
from .evaluation import grouping_changes
from .policy import GroupingPolicy
from .storage import atomic_checkpoint, atomic_json, fingerprint
from .training import load_policy


def _physical_row(adapter):
    return {side: [{"id": int(i), "position": row["position"].tolist(),
                    "velocity": row["velocity"].tolist(), "health": float(row["health"])}
                   for i, row in sorted(adapter.agent_states(side).items())]
            for side in ("Red", "Blue")}


def continue_from_snapshot(env, snapshot, action, seed, *, record_trajectory=False):
    """Execute one action then fixed StaticPolicy continuation to real terminal.

    The complete state includes LCL memory. New continuation randomness is
    independent of the saved official future and equal across paired actions.
    """
    env.restore(snapshot)
    env.set_rng(seed)
    initial_step = env.state().step
    continuation = StaticPolicy()
    trajectory, first_actions = [], None
    original_step = env.adapter.step

    def tracked_step(*args, **kwargs):
        nonlocal first_actions
        result = original_step(*args, **kwargs)
        if first_actions is None:
            first_actions = list(result[3]["red_actions"])
        if record_trajectory:
            trajectory.append({"step": env.adapter.step_count, **_physical_row(env.adapter)})
        return result

    env.adapter.step = tracked_step
    try:
        state, reward, done, info = env.step(action)
        total = float(reward)
        while not done:
            decision = continuation.act(state)
            state, reward, done, info = env.step(decision.action)
            total += float(reward)
    finally:
        env.adapter.step = original_step
    result = {"seed": int(seed), "success": bool(info["success"]), "return": total,
              "first_actions": first_actions, "physical_steps": state.step - initial_step,
              "terminal_step": state.step, "red_survivors": len(state.ids("red")),
              "blue_survivors": len(state.ids("blue"))}
    if record_trajectory:
        result.update(trajectory=trajectory, trajectory_hash=fingerprint(trajectory))
    return result


def same_assignment_partitions(grouping: Grouping) -> tuple[Grouping, Grouping]:
    """Merge/chunk versus singleton groups, holding every task and reserve ID."""
    by_target = {}
    for group in grouping.groups:
        by_target.setdefault(group.target, []).extend(group.members)
    merged, singletons = [], []
    for target, ids in sorted(by_target.items()):
        ids = sorted(ids)
        merged.extend(Group(target, tuple(ids[k:k + 4])) for k in range(0, len(ids), 4))
        singletons.extend(Group(target, (i,)) for i in ids)
    return Grouping(tuple(merged), grouping.reserve), Grouping(tuple(singletons), grouping.reserve)


def _rank_correlation(scores, outcomes):
    if len(scores) < 2 or len(set(scores)) < 2 or len(set(outcomes)) < 2:
        return None
    return float(spearmanr(scores, outcomes).statistic)


def _timed_action(policy, state, *, count, seed):
    durations = {"encoding_seconds": 0.0, "selection_scoring_seconds": 0.0,
                 "repair_scoring_seconds": 0.0}
    originals = {}
    device = next(policy.parameters()).device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for name, key in (("_encode", "encoding_seconds"), ("_selection", "selection_scoring_seconds"),
                      ("_repair", "repair_scoring_seconds")):
        original = getattr(policy, name)
        originals[name] = original

        def measured(*args, _original=original, _key=key, **kwargs):
            synchronize()
            start = time.perf_counter()
            result = _original(*args, **kwargs)
            synchronize()
            durations[_key] += time.perf_counter() - start
            return result

        setattr(policy, name, measured)
    synchronize()
    started = time.perf_counter()
    try:
        with torch.no_grad():
            decision = policy.act(state, deterministic=True, rng=np.random.default_rng(seed), release_count=count)
        synchronize()
        durations["decision_seconds"] = time.perf_counter() - started
    finally:
        for name, original in originals.items():
            setattr(policy, name, original)
    return decision, durations


def _policy(config, checkpoint):
    if checkpoint is not None:
        supplied, payload = load_policy(checkpoint, config.get("device", "cpu"))
        if isinstance(supplied, GroupingPolicy):
            supplied.mode = "selective"
            supplied.eval().requires_grad_(False)
            return supplied, f"checkpoint:{payload['config']['method']}", None
        q_reference = supplied if hasattr(supplied, "q") else None
    else:
        q_reference = None
    model = config.get("model", {})
    kwargs = {key: model[key] for key in ("hidden_dim", "heads", "layers") if key in model}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.get("seed", 0)) + 9301)
        policy = GroupingPolicy(mode="selective", **kwargs)
    policy.to(config.get("device", "cpu")).eval().requires_grad_(False)
    return policy, "untrained_initialization", q_reference


def run_diagnostics(config, output, checkpoint=None, states=3, rollouts=2, wall_seconds=120) -> dict:
    """Run bounded real-world probes and report completed quantities explicitly.

    E2 screens and independently validates every common-pool action using a
    fixed continuation; its screen winner is a conditional reference, never
    an optimal value for the full sequential task. Controlled deaths are
    separately labelled probes, not sampled evaluation outcomes.
    """
    if states < 1 or rollouts < 1 or (wall_seconds is not None and wall_seconds <= 0):
        raise ValueError("states, rollouts and any wall_seconds budget must be positive")
    if int(config.get("max_steps", 50)) != 50:
        raise ValueError("E1–E3 use the frozen DLOM's native 50-step task horizon")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = float("inf") if wall_seconds is None else started + float(wall_seconds)
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    candidate_limit = min(16, max(8, int(config.get("diagnostics", {}).get("candidate_limit", 16))))
    result = {"schema": "known-grouping-diagnostics-v2", "requested_states": int(states),
              "requested_rollouts": int(rollouts), "candidate_limit": candidate_limit,
              "max_steps": int(config.get("max_steps", 50)), "completed_states": 0,
              "completed_physical_rollouts": 0, "completed_physical_steps": 0,
              "state_records": [], "e1": [], "e2": [], "e3": [],
              "limitations": ["Saved controlled-casualty states are diagnostic interventions, not evaluation samples.",
                              "Reference values condition on the current candidate pool and fixed static continuation, not Q*.",
                              "Equal-release-count E3 full is a restricted diagnostic; normal full release is reported separately.",
                              "Selection/repair sub-times cover score computations; total decision time includes encoding, sampling and masks.",
                              "These small probes may find no effect and do not establish algorithm superiority."]}
    base_policy, policy_status, q_reference = _policy(config, checkpoint)
    result["e3_policy_status"] = policy_status
    proxy = DLOMSearchPolicy(device=config.get("executor_device", "cpu"), limit=candidate_limit)
    seed_base = int(config.get("seed", 0)) + 5000000

    def budget():
        return time.perf_counter() < deadline

    def branch(env, snapshot, action, seed, trace=False):
        row = continue_from_snapshot(env, snapshot, action, seed, record_trajectory=trace)
        result["completed_physical_rollouts"] += 1
        result["completed_physical_steps"] += row["physical_steps"]
        return row

    for index in range(states):
        if not budget():
            break
        scale = (4, 8)[index % 2]
        env = KnownOpponentEnv(red=scale, blue=scale, max_steps=result["max_steps"],
                               command_interval=int(config.get("command_interval", 5)),
                               opponent=config.get("opponent", "reactive"),
                               device=config.get("executor_device", "cpu"), seed=seed_base + index)
        try:
            state = env.reset()
            initial = env.snapshot()
            first = balanced_initial(state)
            state, _, done, _ = env.step(first)
            if done:
                state = env.restore(initial)
                # A very short developer horizon may leave only step zero.
                env.previous = first
                state = env.state()
            removed = []
            kind = ("natural", "controlled_light_casualty", "controlled_severe_casualty")[index % 3]
            if kind != "natural":
                alive = [agent for agent in env.adapter.env.red_agents if agent.Health > 0]
                count = min(max(0, len(alive) - 2), 1 if index % 3 == 1 else len(alive) // 2)
                for agent in alive[:count]:
                    agent.Health = 0.0
                    removed.append(int(agent.Id))
                env.adapter.env.update_alive_agents()
                state = env.state()
            env.previous = state.previous
            snapshot = env.snapshot()
            state = env.state()
            state_id = f"state_{index:03d}"
            atomic_checkpoint(output / "states" / f"{state_id}.pt", snapshot)
            atomic_json(output / "states" / f"{state_id}.json", state.to_dict())
            result["state_records"].append({"state_id": state_id, "kind": kind,
                "initial_scale": scale, "remaining_red": len(state.ids("red")), "step": state.step,
                "controlled_removed_ids": removed,
                "state_hash": fingerprint(state.to_dict()), "snapshot": f"states/{state_id}.pt"})
            result["completed_states"] += 1

            # E1: task assignments, environment, memory and continuation match.
            first, second = same_assignment_partitions(state.previous)
            e1 = {"state_id": state_id, "comparable": first != second,
                  "actions": [first.to_dict(), second.to_dict()], "pairs": [],
                  "same_assignment": first.assignment() == second.assignment()}
            if e1["comparable"]:
                for replicate in range(rollouts):
                    if not budget():
                        break
                    seed = seed_base + 10000 + index * 100 + replicate
                    left = branch(env, snapshot, first, seed, True)
                    right = branch(env, snapshot, second, seed, True)
                    path = f"e1/{state_id}_pair_{replicate}.json"
                    atomic_json(output / path, {"left": left, "right": right})
                    e1["pairs"].append({"seed": seed,
                        "first_actions_differ": left["first_actions"] != right["first_actions"],
                        "trajectory_differs": left["trajectory_hash"] != right["trajectory_hash"],
                        "terminal_success_differs": left["success"] != right["success"],
                        "returns": [left["return"], right["return"]], "trajectory_file": path})
            result["e1"].append(e1)

            # E2: a shared, policy-independent action pool; two independent seed sets.
            pool = candidates(state, candidate_limit, np.random.default_rng(seed_base + index))
            scores = proxy.rank(state, pool)
            e2 = {"state_id": state_id, "candidate_count": len(pool), "candidates": [],
                  "proxy_scores": list(map(float, scores)), "complete": False,
                  "continuation": "StaticPolicy: retain chosen groups and remove only dead IDs"}
            screen_seeds = [seed_base + 20000 + index * 100 + i for i in range(rollouts)]
            final_seeds = [seed_base + 30000 + index * 100 + i for i in range(rollouts)]
            e2.update(screen_seeds=screen_seeds, final_seeds=final_seeds)
            for action_index, action in enumerate(pool):
                row = {"index": action_index, "action": action.to_dict(),
                       **grouping_changes(state.previous, action, state.ids("red")),
                       "screen_returns": [], "final_returns": []}
                for label, seeds in (("screen_returns", screen_seeds), ("final_returns", final_seeds)):
                    for seed in seeds:
                        if not budget():
                            break
                        row[label].append(branch(env, snapshot, action, seed)["return"])
                e2["candidates"].append(row)
                if not budget():
                    break
            complete = len(e2["candidates"]) == len(pool) and all(
                len(row[key]) == rollouts for row in e2["candidates"] for key in ("screen_returns", "final_returns"))
            if complete:
                screening = [float(np.mean(row["screen_returns"])) for row in e2["candidates"]]
                final = [float(np.mean(row["final_returns"])) for row in e2["candidates"]]
                reference, selected = int(np.argmax(screening)), int(np.argmax(scores))
                best = max(final)
                high = [row for row, value in zip(e2["candidates"], final) if value >= best - .05]
                e2.update(complete=True, proxy_rank_correlation=_rank_correlation(scores, final),
                          screen_reference_index=reference, proxy_selected_index=selected,
                          independent_reference_return=final[reference], independent_selected_return=final[selected],
                          selected_gap=final[reference] - final[selected], final_returns=final,
                          high_return_candidate_indices=[row["index"] for row in high],
                          high_return_min_task_change=min(row["task_change"] for row in high),
                          high_return_min_team_change=min(row["team_change"] for row in high))
                if q_reference is not None:
                    with torch.no_grad():
                        q_scores = q_reference.q([state] * len(pool), pool).cpu().tolist()
                    e2["alma_action_q_rank_correlation"] = _rank_correlation(q_scores, final)
            result["e2"].append(e2)

            # E3 changes release selection only; every mode has identical weights.
            e3 = {"state_id": state_id, "release_count": max(1, len(state.ids("red")) // 2),
                  "matched_count": [], "normal_release": []}
            if state.step == 0:
                e3["skipped_reason"] = "Initial deployment forces full repair; no noninitial selective decision available."
            else:
                for mode in ("selective", "full", "random", "rule"):
                    if not budget():
                        break
                    policy = copy.deepcopy(base_policy)
                    policy.mode = mode
                    decision, durations = _timed_action(policy, state, count=e3["release_count"],
                                                         seed=seed_base + 40000 + index)
                    row = {"mode": mode, "released_ids": list(decision.released_ids),
                           "action": decision.action.to_dict(), **durations,
                           **grouping_changes(state.previous, decision.action, state.ids("red")), "returns": []}
                    for replicate in range(rollouts):
                        if not budget():
                            break
                        seed = seed_base + 50000 + index * 100 + replicate
                        row["returns"].append(branch(env, snapshot, decision.action, seed)["return"])
                    e3["matched_count"].append(row)
                    if mode in ("full", "selective") and budget():
                        normal, duration = _timed_action(policy, state, count=None, seed=seed_base + 60000 + index)
                        e3["normal_release"].append({"mode": mode, "released_count": len(normal.released_ids),
                            "release_ratio": len(normal.released_ids) / max(1, len(state.ids("red"))),
                            **duration, **grouping_changes(state.previous, normal.action, state.ids("red"))})
            result["e3"].append(e3)
        finally:
            env.close()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["requested_seconds"] = wall_seconds
    result["completed_e1_pairs"] = sum(len(row["pairs"]) for row in result["e1"])
    result["completed_e2_states"] = sum(row["complete"] for row in result["e2"])
    result["completed_e3_modes"] = sum(len(item["returns"]) == rollouts
                                        for row in result["e3"] for item in row["matched_count"])
    result["status"] = ("complete" if result["completed_states"] == states
                        and result["completed_e2_states"] == states
                        and result["completed_e3_modes"] == states * 4 else "partial")
    atomic_json(output / "diagnostics.json", result)
    _write_report(output, result)
    return result


def _write_report(output: Path, result: dict):
    pairs = [pair for row in result["e1"] for pair in row["pairs"]]
    changed_actions = sum(pair["first_actions_differ"] for pair in pairs)
    changed_trajectories = sum(pair["trajectory_differs"] for pair in pairs)
    changed_success = sum(pair["terminal_success_differs"] for pair in pairs)
    lines = ["# 已知完整对手动态分组：E1–E3 诊断", "",
             f"状态：{result['status']}；保存状态 {result['completed_states']}/{result['requested_states']}；"
             f"真实续行 {result['completed_physical_rollouts']} 局、{result['completed_physical_steps']} 物理步。", "",
             "所有分支恢复物理状态、原编组、成员循环记忆与上一动作；续行使用独立随机流。受控减员单独标注。", "",
             f"E1：完成 {len(pairs)} 对。固定任务归属只改变组内关系，首步动作不同 {changed_actions} 对，"
             f"物理轨迹不同 {changed_trajectories} 对，最终成功不同 {changed_success} 对。", "",
             "动作或轨迹差异只能说明编组具有可控影响；最终成功相同也如实保留，不能据此声称胜率提高。", "",
             f"E2：完整评价 {result['completed_e2_states']} 个共同候选池。筛选与独立评价随机种子不重叠，"
             "参考动作由筛选集选出，差距由独立评价集估计；固定 StaticPolicy 续行，因此不是全任务最优 Q*。", ""]
    for row in result["e2"]:
        if row["complete"]:
            lines.append(f"- {row['state_id']}：候选 {row['candidate_count']}，DLOM 排序相关 "
                         f"{row['proxy_rank_correlation']}，独立评价参考减去 DLOM 选择收益 "
                         f"{row['selected_gap']:.3f}；高收益候选最少任务修改 {row['high_return_min_task_change']:.3f}，"
                         f"最少队友关系修改 {row['high_return_min_team_change']:.3f}。")
        else:
            lines.append(f"- {row['state_id']}：预算内未完成全部候选及重复续行，不报告排序结论。")
    lines.extend(["", f"E3：完成 {result['completed_e3_modes']} 个同释放人数比较；模型状态为 {result['e3_policy_status']}。",
                  "同一模型与重建器仅切换释放规则；完整重构的等人数探针有额外限制，正常完整与选择性释放规模单列。",
                  "任务变更与队友变更按真实成员关系计算，不按临时组编号。决策总时含编码、掩码和构造。", "",
                  "此诊断规模不足以证明选择性方法优于完整重构；若 E1 没有测到效果，应先检查执行接口。", "",
                  "逐候选回报、独立种子、真实轨迹、计时与未完成数量见 diagnostics.json 及 states/、e1/。"])
    (output / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["run_diagnostics", "same_assignment_partitions", "continue_from_snapshot"]
