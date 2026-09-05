"""Episode registration/execution with simultaneous hidden Blue commands."""
from __future__ import annotations
from dataclasses import replace
import time
import numpy as np
from open_score.envs import HADStage3Adapter
from open_score.stage3.runtime import moderate_jittered_target_positions
from .domain import action_from_counts, observe, quotas, seed_for
from .planner import ABLATION_METHODS, history_state, tail_action
from .policies import RiskProxy, blue_distribution, sample_blue
from .world import Execution, isolated_rng


def training_registry(config):
    cells = []
    for scenario in config["scenarios"]:
        for lower in config["lower_policies"]:
            for upper in config["closed_upper"]:
                for probe in config["training"]["probes"]:
                    for repeat in range(config["training"]["trajectories_per_cell"]):
                        seed = seed_for("train",config["seed"],scenario["label"],lower,upper,probe,repeat)
                        cells.append({"scenario":scenario,"lower":lower,"upper":upper,"probe":probe,"seed":seed,
                                      "id":f"train_{scenario['label']}_{lower}_{upper}_{probe}_{repeat:03d}"})
    if len({x["seed"] for x in cells}) != len(cells):
        raise ValueError("Training seed collision")
    order = sorted(range(len(cells)),key=lambda i:seed_for("split",cells[i]["seed"]))
    for rank,i in enumerate(order):
        cells[i]["split"] = "train" if rank < .70*len(cells) else "validation" if rank < .85*len(cells) else "test"
    return cells


def formal_registry(config):
    cells = []
    for scenario in config["scenarios"]:
        for lower in config["lower_policies"]:
            for upper in config["closed_upper"]+config["open_upper"]:
                for repeat in range(config["episodes_per_cell"]):
                    seed = seed_for("formal",config["seed"],scenario["label"],lower,upper,repeat)
                    for method in config["methods"]:
                        cells.append({"scenario":scenario,"lower":lower,"upper":upper,"method":method,"repeat":repeat,
                                      "seed":seed,"id":f"formal_{scenario['label']}_{lower}_{upper}_{method}_{repeat:02d}"})
    if len(cells) != config["acceptance"]["expected_episodes"]:
        raise ValueError("Formal registration must contain the complete fixed-type grid")
    if {x["seed"] for x in cells} & {x["seed"] for x in training_registry(config)}:
        raise ValueError("Train/formal seed collision")
    return cells


def make_adapter(cell,config):
    scenario = cell["scenario"]
    positions = moderate_jittered_target_positions(scenario["targets"],config["physical"]["target_layout"],seed=cell["seed"])
    adapter = HADStage3Adapter(scenario["red"],scenario["blue"],scenario["targets"],max_steps=config["physical"]["max_steps"],
                              target_positions=positions,blue_rule_style=cell["lower"])
    adapter.reset(seed=cell["seed"])
    return adapter


def ablation_registry(config):
    """Matched controls extend, rather than replace, the 1,920 core units."""
    if tuple(config["paper_ablations"]["methods"]) != ABLATION_METHODS:
        raise ValueError("Paper controls must isolate filtering and representation")
    cells = []
    for original in formal_registry(config):
        if original["method"] != "qom_mcp":
            continue
        for method in ABLATION_METHODS:
            cell = dict(original,method=method)
            cell["id"] = original["id"].replace("formal_", "ablation_", 1).replace("qom_mcp", method)
            cells.append(cell)
    if len(cells) != config["paper_ablations"]["expected_episodes"]:
        raise ValueError("Incomplete matched ablation grid")
    return cells


def is_event(info, config):
    return bool(info["step"] % config["physical"]["command_interval"]==0 or
                (config["physical"]["replan_on_casualty"] and any(e["kind"]=="agents_destroyed" for e in info["events"])))


def execute_episode(cell,config,predictor,stage1,device,commander=None):
    """The hidden action never enters a deployable Red commander argument.

    Counter-based Blue randomness is shared across Red methods at a physical
    tick. States and event counts can diverge, so actions need not be identical.
    """
    is_training = "probe" in cell
    adapter = make_adapter(cell,config)
    state = observe(adapter)
    target_positions = tuple(x.position for x in state.targets)
    counts = [0.0]*len(state.targets)
    frames = [state.to_dict()] if is_training else []
    commands, logs, surprise, filtered = [],[],[],[]
    last_action = None
    need_plan, done = True,False
    outcome = 0
    event_index = 0
    method = cell.get("method","training_probe")
    belief = None if is_training else commander.new_belief(method,cell["upper"] if method=="known_upper_mcp" else None)
    while not done:
        state = observe(adapter,counts)
        if need_plan:
            proxy = RiskProxy(state,predictor)
            # Neither the Blue policy nor its random stream sees method or a_R.
            blue_rng = np.random.default_rng(seed_for("blue",cell["seed"],cell["scenario"]["label"],cell["lower"],cell["upper"],state.step))
            blue = sample_blue(state,cell["upper"],blue_rng,proxy)
            if is_training:
                red = action_from_counts(state,"red",quotas(len(state.ids("red")),np.ones(len(state.targets)))) if cell["probe"]=="balanced" else tail_action(state)
                commands.append({"public":state.to_dict(),"blue_action":blue.to_dict(),"event_index":event_index})
            else:
                start = time.perf_counter()
                # Opponent inference and planning get a separate RNG namespace.
                plan_rng = np.random.default_rng(seed_for("red-planner",cell["seed"],method,state.step))
                if method == "prior_mcp":
                    # Remove inter-event type learning while preserving public
                    # state, physical reranking and within-search observation use.
                    belief.posterior[:] = 1/len(belief.posterior)
                belief.prepare(state,proxy,plan_rng)
                probabilities = belief.targets(state)
                red,diagnostics = commander.decide(state,method,belief,plan_rng,last_action,
                                                   revealed=blue if method=="revealed_blue_action_br" else None)
                diagnostics["planning_seconds"] = time.perf_counter()-start
                # Scoring/logging is downstream of decision; private labels are
                # never passed back into belief.update or model.forward.
                target_index = {x.id:i for i,x in enumerate(state.targets)}
                labels = [target_index[blue.assignment()[i]] for i in state.ids("blue")]
                predicted = probabilities.argmax(-1).tolist() if labels else []
                forecast_ceiling = None
                if method=="qom_mcp":
                    # Evaluator-only, after Red commits. This is an expectation
                    # over the known program, not knowledge of its current draw.
                    oracle_actions,oracle_weights = blue_distribution(state,cell["upper"],proxy)
                    oracle_probabilities = np.zeros_like(probabilities)
                    for action,weight in zip(oracle_actions,oracle_weights):
                        mapping = action.assignment()
                        for i,agent_id in enumerate(state.ids("blue")):
                            oracle_probabilities[i,target_index[mapping[agent_id]]] += weight
                    forecast_ceiling = float(oracle_probabilities.max(-1).mean()) if labels else None
                logs.append({"step":state.step,"event_index":event_index,"red":red.to_dict(),"blue":blue.to_dict(),
                             "posterior":belief.posterior.tolist(),"truth_targets":labels,"predicted_targets":predicted,
                             "target_probabilities":probabilities.tolist(),"alive_red":len(state.ids("red")),
                             "alive_blue":len(state.ids("blue")),"forecast_oracle_accuracy_ceiling":forecast_ceiling,**diagnostics})
            red.validate(state.ids("red"),[x.id for x in state.targets])
            blue.validate(state.ids("blue"),[x.id for x in state.targets],"Blue")
            execution = Execution(adapter,state,red,blue,stage1,device,config["physical"]["reserve_mode"])
            last_action = red
            event_index += 1
        # Reset only the independent environment continuation stream, not the
        # scenario/layout stream. Planner branch calls restore np.random state.
        with isolated_rng(seed_for("physical",cell["seed"],adapter.step_count)):
            _,_,done,info = execution.step(adapter)
        assigned = last_action.counts([x.id for x in state.targets])
        counts = [old+new for old,new in zip(counts,assigned)]
        after = observe(adapter,counts)
        if tuple(x.position for x in after.targets) != target_positions:
            raise RuntimeError("Fixed targets moved during evaluation")
        if is_training:
            frames.append(after.to_dict())
        else:
            nll = belief.update(state,after)
            surprise.append({"step":after.step,"nll":nll,"unknown_weight":belief.unknown_weight})
            current = belief.current_targets(after)
            labels = [target_index[blue.assignment()[i]] for i in after.ids("blue")]
            filtered.append({"step":after.step,"event_index":event_index-1,"truth_targets":labels,
                             "predicted_targets":current.argmax(-1).tolist() if labels else []})
        outcome = int(info["outcome_red"])
        need_plan = not done and is_event(info,config)
    result = {"cell":cell,"red_win":int(outcome>0),"steps":adapter.step_count,"outcome":outcome,
              "fixed_targets":True,"identity_valid":True,"infeasible":False}
    if is_training:
        result.update({"frames":frames,"commands":commands})
    else:
        result.update({"events":logs,"surprise":surprise,"filtered_assignments":filtered})
    return result
