"""Matched six-method physical evaluation, with public-only planner capabilities."""
import time
import numpy as np
from .domain import observe, seed_for
from .evaluation import make_adapter, is_event
from .fast_planner import FastCommander, METHODS
from .policies import RiskProxy, sample_blue
from .world import Execution, isolated_rng


def registry(config):
    cells = []
    for repeat in range(config["episodes_per_cell"]):
        for scenario in config["scenarios"]:
            for lower in config["lower_policies"]:
                for upper in config["closed_upper"]:
                    seed = seed_for("fast-paired-v2", config["seed"], scenario["label"], lower, upper, repeat)
                    for method in METHODS:
                        cells.append({"id":f"fast_{scenario['label']}_{lower}_{upper}_{repeat:02d}_{method}",
                                      "scenario":scenario, "lower":lower, "upper":upper,
                                      "repeat":repeat, "seed":seed, "method":method})
    return cells


def execute(cell, config, predictor, stage1, device):
    adapter = make_adapter(cell, config)
    commander = FastCommander(predictor, stage1, device, config["planning"])
    method = cell["method"]
    belief = commander.new_belief(method, cell["upper"] if method.startswith("known_") else None)
    infer = method not in {"balanced", "legacy"}
    counts = [0.0]*len(adapter.target_ids)
    targets = tuple(tuple(x.position) for x in adapter.env.targets)
    events, filtered = [], []
    need_plan, done, previous = True, False, None
    event_index, filter_seconds = 0, 0.0
    started = time.perf_counter()
    while not done:
        state = observe(adapter, counts)
        if need_plan:
            # Real Blue always uses the same original five-step proxy/program.
            # Neither method name nor current Red command enters this stream.
            blue_rng = np.random.default_rng(seed_for("blue", cell["seed"], state.step))
            blue = sample_blue(state, cell["upper"], blue_rng, RiskProxy(state, predictor))
            plan_rng = np.random.default_rng(seed_for("red", cell["seed"], method, state.step))
            red, diagnostics = commander.plan(state, method, belief, plan_rng, previous)
            red.validate(state.ids("red"), [t.id for t in state.targets], blue_ids=state.ids("blue"))
            blue.validate(state.ids("blue"), [t.id for t in state.targets], "Blue")
            event = {"step":state.step, "event_index":event_index, "red":red.to_dict(), "blue":blue.to_dict(),
                     "alive_red":len(state.ids("red")), "alive_blue":len(state.ids("blue")), **diagnostics}
            # Current truth is used only after Red has committed.
            if infer:
                target_index = {t.id:i for i,t in enumerate(state.targets)}
                event["truth_targets"] = [target_index[blue.assignment()[i]] for i in state.ids("blue")]
                event["predicted_targets"] = belief.targets(state).argmax(-1).tolist()
            events.append(event)
            execution = Execution(adapter, state, red, blue, stage1, device, config["physical"]["reserve_mode"])
            previous = red
            event_index += 1
        with isolated_rng(seed_for("physical", cell["seed"], state.step)):
            _, _, done, info = execution.step(adapter)
        assigned = previous.counts([t.id for t in state.targets])
        counts = [a+b for a,b in zip(counts,assigned)]
        after = observe(adapter, counts)
        if tuple(t.position for t in after.targets) != targets:
            raise AssertionError("Protected targets moved")
        if infer:
            tick = time.perf_counter()
            belief.update(state, after)
            probabilities = belief.current_targets(after)
            filter_seconds += time.perf_counter()-tick
            filtered.append({"step":after.step, "event_index":event_index-1,
                             "truth_targets":[target_index[blue.assignment()[i]] for i in after.ids("blue")],
                             "predicted_targets":probabilities.argmax(-1).tolist() if after.ids("blue") else []})
        need_plan = not done and is_event(info, config)
    return {"cell":cell, "red_win":int(info["outcome_red"]>0), "steps":adapter.step_count,
            "events":events, "filtered_assignments":filtered, "wall_seconds":time.perf_counter()-started,
            "filter_seconds":filter_seconds, "identity_valid":True, "fixed_targets":True, "infeasible":False}
