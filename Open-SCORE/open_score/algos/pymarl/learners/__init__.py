from .q_learner import QLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner

from open_score.algos.transfqmix import TransfQMixLearner
REGISTRY["transfqmix_learner"] = TransfQMixLearner
