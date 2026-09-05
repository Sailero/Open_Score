"""Common candidate screening, physical reranking and belief-history PUCT.

Finite search is approximate. Only qom_mcp and known_upper_mcp share the full
planning algorithm/budget; baseline wall times are measured, never padded.
"""
from __future__ import annotations
from dataclasses import replace
import time
import numpy as np
from open_score.stage3.blotto import solve_restricted_matrix_game
from .belief import OpponentBelief
from .domain import IdentityUpperAction, action_from_counts, eta, quotas, seed_for
from .policies import RiskProxy, decode_action, sample_blue, shared_candidates, simple_blue_actions
from .world import branch, from_public

METHODS = ("balanced_identity","legacy_idb","event_risk_idb","finite_type_bbr",
           "qom_bbr","qom_mcp","known_upper_mcp","revealed_blue_action_br")
ABLATION_METHODS = ("finite_type_mcp", "prior_mcp")
MCP_METHODS = ("qom_mcp", "known_upper_mcp", *ABLATION_METHODS)


def public_potential(state, outcome):
    if outcome:
        return float(outcome)
    target = min(x.health/1.2 for x in state.targets)
    removed = 1-sum(x.health for x in state.blue)/len(state.blue)
    return float(np.clip(.8*target+.2*removed,0,1))


def history_state(state, red, elapsed):
    counts = red.counts([x.id for x in state.targets])
    return replace(state,red_history_counts=tuple(old+elapsed*c for old,c in zip(state.red_history_counts,counts)))


def tail_action(state):
    threats = [sum(1/(1+eta(b,t)) for b in state.alive("blue"))+.05 for t in state.targets]
    return action_from_counts(state,"red",quotas(len(state.ids("red")),threats))


def observed_key(state):
    # An explicit public observation abstraction, with alive IDs preserved.
    # No theta, sampled action, posterior argmax or hidden simulator state.
    return (state.step, tuple((x.id, x.alive, *(round(v/250) for v in x.position),
                               *(round(v/75) for v in x.velocity),round(x.health*4))
                             for side in [state.red,state.blue,state.targets] for x in side))


class Commander:
    def __init__(self,predictor,stage1,device,config,qom_factory=None):
        self.predictor,self.stage1,self.device,self.config = predictor,stage1,device,config
        self.qom_factory = qom_factory
        self.total_branch_steps = 0

    def new_belief(self,method,known_type=None):
        if method in {"qom_bbr","qom_mcp"}:
            return OpponentBelief("qom",qom=self.qom_factory())
        return OpponentBelief(known_type=known_type if method=="known_upper_mcp" else None)

    def simulate(self,state,red,blue,belief,seed,*,reserve_mode="patrol"):
        nxt,outcome,trace = branch(state,red,blue,self.stage1,self.device,seed,reserve_mode=reserve_mode)
        self.total_branch_steps += len(trace)-1
        for before,after in zip(trace,trace[1:]):
            belief.update(before,after)
        nxt = history_state(nxt,red,nxt.step-state.step)
        return nxt,outcome

    def latent_action(self,state,theta,belief,proxy,rng):
        if belief.kind=="qom":
            p,g = belief.event_distributions
            return decode_action(state,p[int(theta)],g[int(theta)],rng)
        return sample_blue(state,theta,rng,proxy)

    def rollout(self,state,theta,belief,rng):
        while state.step < state.max_steps and state.ids("blue") and all(x.health>=1e-3 for x in state.targets):
            proxy = RiskProxy(state,self.predictor)
            belief.prepare(state,proxy,rng)
            blue = self.latent_action(state,theta,belief,proxy,rng)
            red = tail_action(state)
            nxt,outcome = self.simulate(state,red,blue,belief,int(rng.integers(2**32)))
            if outcome:
                return float(outcome)
            if nxt.step <= state.step:
                raise RuntimeError("Rollout did not advance physical time")
            state = nxt
        return -1.0 if any(x.health<1e-3 for x in state.targets) else 1.0

    def legacy(self,state,rng):
        from open_score.stage3.identity_runtime import build_identity_event_game
        from open_score.stage3.identity_blotto import solve_identity_double_oracle
        built = build_identity_event_game(from_public(state,0),self.predictor,blue_style=state.lower,
                                          max_group_size=4,full_domain_agent_threshold=8,
                                          utility_mode="joint_survival_log_probability",risk_epsilon=1e-6)
        result = solve_identity_double_oracle(built.game,built.initial_red,built.initial_blue,
                                              tolerance=.01,max_iterations=12,oracle_time_limit_seconds=1.0,
                                              oracle_mip_relative_gap=.01)
        action,_ = result.sample_profile(rng=rng)
        return IdentityUpperAction(tuple((slot.target_id,ids) for slot,ids in zip(built.slots,action.coalitions) if ids),action.reserve_ids)

    def decide(self,state,method,belief,rng,previous=None,revealed=None):
        if method not in METHODS + ABLATION_METHODS:
            raise ValueError(f"Unregistered Red method: {method}")
        start = time.perf_counter()
        branch_start = self.total_branch_steps
        if not state.ids("red"):
            return IdentityUpperAction(()),{"planning_seconds":0.,"candidate_count":1,"branch_steps":0}
        proxy = RiskProxy(state,self.predictor)
        candidates,blue_domain,matrix = shared_candidates(state,proxy,previous,self.config["candidates"])
        signature = seed_for([x.to_dict() for x in candidates])
        if method == "balanced_identity":
            chosen = candidates[0]
        elif method == "legacy_idb":
            chosen = self.legacy(state,rng)
        else:
            equilibrium = solve_restricted_matrix_game(matrix)
            minimax = np.asarray(equilibrium.defender_mixture)
            if method == "event_risk_idb":
                probabilities = np.asarray(equilibrium.attacker_mixture)
                scenarios = [(None,blue_domain[int(rng.choice(len(blue_domain),p=probabilities))]) for _ in range(self.config["rerank_repeats"])]
            elif method == "revealed_blue_action_br":
                if revealed is None:
                    raise ValueError("Revealed baseline needs an explicit current-action capability")
                scenarios = [(None,revealed)]*self.config["rerank_repeats"]
            else:
                scenarios = [belief.sample(rng,state) for _ in range(self.config["rerank_repeats"])]
            # Same sampled opponent actions and random continuations per candidate.
            branch_seeds = rng.integers(0,2**32,len(scenarios),dtype=np.uint64)
            scores = np.zeros(len(candidates))
            for i,red in enumerate(candidates):
                for j,(_,blue) in enumerate(scenarios):
                    nxt,outcome,trace = branch(state,red,blue,self.stage1,self.device,int(branch_seeds[j]),stop_on_event=False)
                    self.total_branch_steps += len(trace)-1
                    scores[i] += public_potential(nxt,outcome)/len(scenarios)
            if method in MCP_METHODS:
                chosen = self.mcts(state,candidates,scores,belief,rng)
            elif method == "event_risk_idb":
                # Reweight the minimax support with actual shared-world outcomes.
                weights = minimax*np.exp(3*(scores-scores.max()))
                chosen = candidates[int(rng.choice(len(candidates),p=weights/weights.sum()))]
            else:
                # Frozen Stage2 posterior value breaks physical short-window ties.
                posterior_values = (proxy.values(candidates,[revealed])[:,0]
                                    if method == "revealed_blue_action_br" else
                                    proxy.values(candidates,belief.actions) @ belief.weights)
                index = max(range(len(candidates)),key=lambda k:(scores[k],posterior_values[k],-k))
                chosen = candidates[index]
            if method in {"qom_bbr","qom_mcp"} and rng.random() < belief.unknown_weight:
                chosen = candidates[int(rng.choice(len(candidates),p=minimax))]
        chosen.validate(state.ids("red"),[x.id for x in state.targets],blue_ids=state.ids("blue"))
        return chosen,{"planning_seconds":time.perf_counter()-start,"candidate_count":len(candidates),
                       "candidate_signature":signature,"branch_steps":self.total_branch_steps-branch_start,
                       "puct_simulations":self.config["simulations"] if method in MCP_METHODS else 0,
                       "unknown_weight":belief.unknown_weight}

    def mcts(self,root,candidates,screening,root_belief,rng):
        prior = np.exp(2*(screening-screening.max()))+.01
        prior /= prior.sum()
        tree = {}
        root_key = ((),observed_key(root))
        tree[root_key] = [candidates,np.zeros(len(candidates)),np.zeros(len(candidates)),prior]
        for simulation in range(self.config["simulations"]):
            belief = root_belief.copy()
            theta,blue = belief.sample(rng,root)
            state,path,ancestors = root,[],()
            outcome = 0
            for depth in range(self.config["depth"]):
                key = (ancestors,observed_key(state))
                if key not in tree:
                    proxy = RiskProxy(state,self.predictor)
                    actions,_,_ = shared_candidates(state,proxy,limit=self.config["candidates"])
                    values = proxy.values(actions,belief.actions) @ belief.weights
                    p = np.exp(np.clip(values-values.max(),-20,0))+.01
                    tree[key] = [actions,np.zeros(len(actions)),np.zeros(len(actions)),p/p.sum()]
                actions,n,w,p = tree[key]
                if (n==0).any():
                    choice = int(np.flatnonzero(n==0)[0])
                else:
                    choice = int(np.argmax(w/np.maximum(n,1)+1.4*p*np.sqrt(n.sum()+1)/(1+n)))
                red = actions[choice]
                nxt,outcome = self.simulate(state,red,blue,belief,int(rng.integers(2**32)))
                path.append((key,choice))
                ancestors += (choice,)
                state = nxt
                if outcome:
                    break
                if depth+1 < self.config["depth"]:
                    proxy = RiskProxy(state,self.predictor)
                    belief.prepare(state,proxy,rng)
                    # Persistent theta/code across the entire simulated episode.
                    blue = self.latent_action(state,theta,belief,proxy,rng)
            value = float(outcome) if outcome else self.rollout(state,theta,belief,rng)
            for key,choice in path:
                tree[key][1][choice] += 1
                tree[key][2][choice] += value
        actions,n,w,_ = tree[root_key]
        return actions[max(range(len(actions)),key=lambda i:(n[i],w[i]/max(n[i],1),-i))]
