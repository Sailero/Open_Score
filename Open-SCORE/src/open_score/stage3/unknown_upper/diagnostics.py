"""Pre-formal checks; empirical diagnostics are retained even when negative."""
from __future__ import annotations
from dataclasses import replace
import itertools
import time
import numpy as np
from scipy.stats import spearmanr
from open_score.stage3.blotto import solve_restricted_matrix_game
from .belief import OpponentBelief
from .domain import Entity,IdentityUpperAction,PublicState,canonical_blue,match,observe,red_candidates,seed_for
from .evaluation import make_adapter,training_registry
from .planner import public_potential
from .policies import CLOSED_TYPES,RiskProxy,blue_distribution,shared_candidates,simple_blue_actions
from .storage import atomic_json,read_unit
from .world import branch,from_public,install,predicted_velocities


def enumerate_actions(ids,targets,allow_reserve=False,opponent_ids=()):
    result = set()
    def recurse(todo,groups,reserves):
        if not todo:
            result.add(IdentityUpperAction(tuple(groups),tuple(reserves)))
            return
        first,*rest = todo
        if allow_reserve:
            recurse(rest,groups,reserves+[first])
        for target in targets:
            for size in range(min(3,len(rest))+1):
                for peers in itertools.combinations(rest,size):
                    recurse([i for i in rest if i not in peers],groups+[(target,(first,*peers))],reserves)
    recurse(list(ids),[],[])
    if opponent_ids:
        extended=set()
        for action in result:
            own=[ids for _,ids in action.groups]
            for choices in itertools.product(range(-1,len(own)),repeat=len(opponent_ids)):
                intents=tuple((r,tuple(i for i,c in zip(opponent_ids,choices) if c==k)) for k,r in enumerate(own) if k in choices)
                if all(len(b)<=4 for _,b in intents):
                    extended.add(IdentityUpperAction(action.groups,action.reserve_ids,intents))
        result=extended
    return tuple(sorted(result,key=lambda a:(a.groups,a.reserve_ids,a.intercepts)))


def exact_check(predictor):
    state = PublicState(10,50,"split_rush",(Entity(0,(-1700.,-40.,100.),(20.,0.,0.),1.),Entity(1,(-1800.,50.,100.),(20.,0.,0.),1.)),
                        (Entity(2,(-1500.,-60.,100.),(-100.,0.,0.),1.),Entity(3,(-1400.,70.,100.),(-100.,0.,0.),1.)),
                        (Entity(0,(-2100.,0.,100.),(0.,0.,0.),1.2),),(0.,))
    # IDs must match HAD's actual allocation, whose target precedes agents.
    adapter = from_public_ids_example(state)
    state = observe(adapter)
    red = enumerate_actions(state.ids("red"),[x.id for x in state.targets],True,state.ids("blue"))
    blue = enumerate_actions(state.ids("blue"),[x.id for x in state.targets])
    matrix = RiskProxy(state,predictor).values(red,blue)
    full = solve_restricted_matrix_game(matrix)
    ri,bi = [0],[0]
    for _ in range(len(red)+len(blue)+1):
        solution = solve_restricted_matrix_game(matrix[np.ix_(ri,bi)])
        br = int(np.argmax(matrix[:,bi]@solution.attacker_mixture))
        bb = int(np.argmin(solution.defender_mixture@matrix[ri,:]))
        if br in ri and bb in bi:
            break
        ri = sorted(set(ri+[br]));bi = sorted(set(bi+[bb]))
    error = abs(full.value-solution.value)
    if error > 1e-7:
        raise AssertionError("Exhaustive vs restricted-oracle solution mismatch")
    return {"red_actions":len(red),"blue_actions":len(blue),"value_error":error,
            "scope":"complete 2v2 single-target upper-action game only; no large-domain certificate"}


def from_public_ids_example(state):
    from open_score.envs import HADStage3Adapter
    adapter = HADStage3Adapter(2,2,1,max_steps=50,target_positions=[state.targets[0].position],blue_rule_style=state.lower)
    adapter.reset(seed=765)
    for agents,records in [(adapter.env.red_agents,state.red),(adapter.env.blue_agents,state.blue)]:
        for agent,row in zip(agents,records):
            agent.position=list(row.position);agent.velocity=list(row.velocity);agent.Health=row.health
    adapter.step_count=state.step
    return adapter


def information_ladder(matrix,labels,counts,weights):
    """Expected optimal proxy values on nested sigma-fields.

    Known type -> known type AND target counts -> current action. Knowing
    counts alone is a separate branch and need not dominate known type.
    """
    weights = np.asarray(weights)/np.sum(weights)
    def conditional(keys):
        value = 0.0
        for key in set(keys):
            indices = [i for i,k in enumerate(keys) if k==key]
            value += float((matrix[:,indices]@weights[indices]).max())
        return value
    unknown = float((matrix@weights).max())
    typ = conditional(labels)
    type_counts = conditional(list(zip(labels,counts)))
    action = float(matrix.max(0)@weights)
    if not unknown <= typ+1e-8 or not typ <= type_counts+1e-8 or not type_counts <= action+1e-8:
        raise AssertionError("Nested information value decreased")
    return {"unknown":unknown,"type":typ,"type_and_counts":type_counts,"full_action":action,
            "counts_only_separate_branch":conditional(counts),"metric":"expected best frozen proxy on one common finite action domain"}


def run_diagnostics(config,root,predictor,stage1,device,commander,training_paths,log):
    result = {"exact":exact_check(predictor),"snapshots":[],"timing":[],"hard_checks_passed":True}
    if not config["diagnostics"].get("extended",False):
        result["extended_diagnostics"] = "not_run: fixed-type experiment prioritizes formal outcomes"
        result["hard_checks_scope"] = "public-interface unit checks, exhaustive small game; runtime identity/target assertions in every formal episode"
        atomic_json(root/"diagnostics.json",result)
        log("Essential constraints checked; extended branch/calibration/timing probes skipped")
        return result
    episodes = [read_unit(p) for p in training_paths]
    validation = [e for e in episodes if e and e["cell"]["split"]=="validation"]
    for scenario in config["scenarios"]:
        for lower in config["lower_policies"]:
            choices = [e for e in validation if e["cell"]["scenario"]["label"]==scenario["label"] and e["cell"]["lower"]==lower]
            for index,episode in enumerate(choices[:config["diagnostics"]["snapshots_per_scale_style"]]):
                # One post-initial and one later, in-distribution public state.
                k = min(1,len(episode["commands"])-1) if index==0 else max(0,len(episode["commands"])-2)
                state = PublicState.from_dict(episode["commands"][k]["public"])
                if not state.ids("red") or not state.ids("blue"):
                    continue
                proxy = RiskProxy(state,predictor)
                reds,_,_ = shared_candidates(state,proxy)
                blues,labels,weights = [],[],[]
                for theta in CLOSED_TYPES:
                    actions,p = blue_distribution(state,theta,proxy)
                    blues.extend(actions);labels.extend([theta]*len(actions));weights.extend(p/len(CLOSED_TYPES))
                matrix = proxy.values(reds,blues)
                expanded=tuple(dict.fromkeys((*reds,*red_candidates(state,limit=64))))
                expanded_matrix=proxy.values(expanded,blues)
                counts = [a.counts([x.id for x in state.targets]) for a in blues]
                ladder = information_ladder(matrix,labels,counts,weights)
                adapter = from_public(state,99)
                old_public = observe(adapter,state.red_history_counts)
                adapter.set_assignments("Blue",{i:state.targets[-1].id for i in adapter.blue_ids})
                if observe(adapter,state.red_history_counts) != old_public:
                    raise AssertionError("Hidden Blue commands leaked through public view")
                a = simple_blue_actions(state,"balanced")[0][0]
                predicted = predicted_velocities(state,a)
                install(adapter,state,reds[0],a)
                commands = adapter.commanded_rule_actions(subgroup_by_agent={i:g for g,(_,ids) in enumerate(a.groups) for i in ids})
                for agent,command in zip(adapter.env.blue_agents,commands):
                    if agent.Health>0:
                        agent.acceleration=(adapter._decode_acceleration(command)*agent.aMax).tolist()
                        agent.update_velocity()
                        if not np.allclose(agent.velocity,predicted[agent.Id],atol=1e-7):
                            raise AssertionError("Public motion model differs from known HAD lower dynamics")
                physical,risk,brier = [],[],[]
                for red_index,red in enumerate(reds):
                    rewards,breaches = [],[]
                    for repeat in range(config["diagnostics"]["branch_repeats"]):
                        rng = np.random.default_rng(seed_for("diagnostic",episode["cell"]["seed"],repeat))
                        b_index=int(rng.choice(len(blues),p=np.asarray(weights)/np.sum(weights)))
                        b = blues[b_index]
                        nxt,outcome,_ = branch(state,red,b,stage1,device,int(rng.integers(2**32)),steps=5,stop_on_event=False)
                        rewards.append(public_potential(nxt,outcome));breaches.append(int(outcome<0))
                        predicted_risk=1-float(np.exp(matrix[red_index,b_index]*len(state.targets)))
                        brier.append((predicted_risk-int(outcome<0))**2)
                    physical.append(float(np.mean(rewards)))
                    risk.append(float(np.mean(breaches)))
                predicted_values = matrix@np.asarray(weights)
                correlation = float(spearmanr(predicted_values,physical).statistic) if np.std(predicted_values)>1e-12 and np.std(physical)>1e-12 else None
                reserve = IdentityUpperAction((),state.ids("red"))
                ablation = {}
                for mode in ["inert","patrol"]:
                    values = []
                    for repeat in range(4):
                        nxt,outcome,_ = branch(state,reserve,a,stage1,device,seed_for("reserve",repeat),reserve_mode=mode,stop_on_event=False)
                        values.append(public_potential(nxt,outcome))
                    ablation[mode]=float(np.mean(values))
                result["snapshots"].append({"scenario":scenario["label"],"lower":lower,"step":state.step,"information":ladder,
                                             "proxy_values":predicted_values.tolist(),"physical_values":physical,"five_step_breach_rates":risk,
                                             "rank_spearman":correlation,"five_step_proxy_brier":float(np.mean(brier)),
                                             "expanded_candidate_count":len(expanded),
                                             "expanded_known_action_proxy_gain":float((expanded_matrix.max(0)-matrix.max(0))@np.asarray(weights)),
                                             "reserve_ablation":ablation,"public_motion_model_matches":True})
                log(f"Diagnostics {scenario['label']}/{lower} snapshot {index+1}: Spearman={correlation}")
    for scenario in config["scenarios"]:
        for repeat in range(config["diagnostics"]["timing_repeats"]):
            cell={"scenario":scenario,"lower":"rush","upper":"balanced","seed":seed_for("timing",config["seed"],scenario["label"],repeat)}
            state=observe(make_adapter(cell,config))
            proxy=RiskProxy(state,predictor)
            rng=np.random.default_rng(seed_for("timing-planner",repeat))
            belief=commander.new_belief("qom_mcp")
            start=time.perf_counter();belief.prepare(state,proxy,rng)
            action,diagnostics=commander.decide(state,"qom_mcp",belief,rng)
            action.validate(state.ids("red"),[x.id for x in state.targets])
            result["timing"].append({"population":scenario["red"]+scenario["blue"],"repeat":repeat,"total_seconds":time.perf_counter()-start,**diagnostics})
            log(f"Timing {scenario['label']}: {result['timing'][-1]['total_seconds']:.2f}s; full PUCT/terminal rollout budget")
    atomic_json(root/"diagnostics.json",result)
    return result
