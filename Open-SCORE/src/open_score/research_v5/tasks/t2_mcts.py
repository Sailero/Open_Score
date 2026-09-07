"""MCTS-DPW with the fixed main evaluation budget."""
from ..planning import MCTSPolicy
from .t1_rollout import load_historical_controls

def run(ctx):
    frozen, _ = load_historical_controls(ctx.config)
    planner = MCTSPolicy(seed=ctx.seed, b1=frozen.get('frozen_b1'), **ctx.config['t2'])
    result = ctx.evaluate(planner, 'T2_mcts_dpw')
    return dict(complete=result['complete'], evaluation=result)
