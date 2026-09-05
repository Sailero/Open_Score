"""Incremental public-motion filtering over persistent types and macro actions."""
from __future__ import annotations
import copy
import numpy as np
from scipy.special import logsumexp
from .domain import prune
from .policies import CLOSED_TYPES, blue_distribution, decode_action
from .world import motion_log_likelihood


class OpponentBelief:
    def __init__(self, kind="finite", qom=None, known_type=None, robust_fallback=False):
        self.kind, self.qom, self.known_type = kind,qom,known_type
        self.robust_fallback = robust_fallback
        self.types = (known_type,) if known_type else (tuple(range(qom.model.codes)) if kind=="qom" else CLOSED_TYPES)
        self.posterior = qom.prior.copy() if kind=="qom" else np.full(len(self.types),1/len(self.types))
        self.actions, self.labels, self.weights = [],[],np.empty(0)
        self.last_nll = 0.0
        self.unknown_weight = 0.0
        self.event_distributions = None
        self.public_updates = 0
        self.last_update_step = None

    def copy(self):
        result = copy.copy(self)
        result.posterior, result.weights = self.posterior.copy(),self.weights.copy()
        result.actions, result.labels = list(self.actions),list(self.labels)
        result.qom = self.qom.copy() if self.qom else None
        return result

    def prepare(self,state,proxy,rng,particles_per_code=4):
        if self.last_update_step is not None and self.last_update_step != state.step:
            raise ValueError("Unobserved gap in public belief history")
        self.last_update_step = state.step
        self.actions,self.labels,weights = [],[],[]
        distributions = self.qom.distributions(state) if self.kind=="qom" else None
        self.event_distributions = distributions
        for k,theta in enumerate(self.types):
            if distributions is not None:
                actions = [decode_action(state,distributions[0][k],distributions[1][k],rng) for _ in range(particles_per_code)]
                probabilities = np.full(len(actions),1/len(actions))
            else:
                actions, probabilities = blue_distribution(state,theta,proxy)
            self.actions.extend(actions)
            self.labels.extend([k]*len(actions))
            weights.extend(self.posterior[k]*probabilities)
        self.weights = np.asarray(weights,float)
        self.weights /= self.weights.sum()
        if distributions is not None and self.robust_fallback:
            concentration = float((distributions[0]*self.posterior[:,None,None]).sum(0).max(-1).mean()) if state.ids("blue") else 1.0
            # Ambiguity is not automatically an unknown policy label.
            self.unknown_weight = max(self.unknown_weight, .5 if concentration < .55 else 0.0)

    def update(self,before,after):
        if after.step != before.step+1:
            raise ValueError("Likelihood must consume each physical transition exactly once")
        if before.step != self.last_update_step:
            raise ValueError("Repeated or skipped public transition")
        if not self.actions:
            raise ValueError("prepare must precede a belief update")
        likelihood = np.asarray([motion_log_likelihood(before,after,a) for a in self.actions])
        log_weights = np.log(np.maximum(self.weights,1e-300)) + likelihood
        evidence = float(logsumexp(log_weights))
        self.last_nll = -evidence
        self.weights = np.exp(log_weights-evidence)
        self.posterior = np.bincount(self.labels,weights=self.weights,minlength=len(self.types))
        self.posterior = np.maximum(self.posterior,1e-8)
        self.posterior /= self.posterior.sum()
        self.public_updates += 1
        self.last_update_step = after.step
        if self.kind=="qom" and self.robust_fallback:
            key = f"{before.lower}:{len(before.red)+len(before.blue)}"
            threshold = float(self.qom.thresholds.get(key,self.qom.thresholds.get("global",50.0)))
            self.unknown_weight = .5 if self.last_nll > threshold else self.unknown_weight*.8
        return self.last_nll

    def sample(self,rng,state=None):
        i = int(rng.choice(len(self.actions),p=self.weights))
        action = self.actions[i]
        return self.types[self.labels[i]], prune(action,state.ids("blue")) if state else action

    def targets(self,state):
        """Forecast after prepare; QOM uses its full decoded target marginal."""
        if self.kind=="qom" and self.event_distributions is not None:
            return (self.event_distributions[0]*self.posterior[:,None,None]).sum(0)
        return self.current_targets(state)

    def current_targets(self,state):
        """Filter the persistent current action with all consumed transitions.

        Unlike next-command prediction this must retain within-code particle
        weights: recent motion may identify an action without identifying code.
        """
        ids = state.ids("blue")
        targets = {x.id:k for k,x in enumerate(state.targets)}
        probabilities = np.zeros((len(ids),len(targets)))
        for action,weight in zip(self.actions,self.weights):
            assignments = action.assignment()
            for k,i in enumerate(ids):
                if i in assignments:
                    probabilities[k,targets[assignments[i]]] += weight
        return probabilities/np.maximum(probabilities.sum(-1,keepdims=True),1e-12)
