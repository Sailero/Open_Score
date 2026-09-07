"""Terminal rollout and the shared fixed baselines."""
from pathlib import Path
import time
from open_score.research_v4.policies import RulePolicy, make_policy
from ..planning import RolloutPolicy, singleton_grouping
from ..protocol import ROOT

class SingletonPolicy:
    last_trace = {}
    def act(self, state):
        return singleton_grouping(state)
    def reset(self):
        self.last_trace = {}

def historical_assets(config):
    return {key: Path(value) for key, value in config.get('frozen_assets', {}).items()}

def load_historical_controls(config):
    policies, metadata = {}, {}
    for method, path in historical_assets(config).items():
        if not path.is_absolute(): path = ROOT/path
        if not path.is_file(): raise FileNotFoundError(f'Missing retained baseline: {path}')
        route = 'b1_counts' if method == 'frozen_b1' else 'b3_global'
        kind = 'count_checkpoint' if method == 'frozen_b1' else 'global_checkpoint'
        policies[method] = make_policy(route, {kind: path}, device='cpu', search_budget=64)
        metadata[method] = dict(artifact=str(path), original_protocol='v4')
    return policies, metadata

def timed_phase(ctx, phase, function):
    started = time.perf_counter()
    result = function()
    ctx.log('phase_costs', dict(phase=phase, wall_seconds=time.perf_counter()-started, episodes=result.get('episodes', 0)))
    return result

def run(ctx):
    frozen, _ = load_historical_controls(ctx.config)
    if ctx.config.get('selected_method') == 'baselines':
        controls = dict(rule=RulePolicy('rule'), grand=RulePolicy('grand'), singleton=SingletonPolicy(), **frozen)
        results = {method: ctx.evaluate(policy, method, checkpoint='frozen') for method, policy in controls.items()}
        return dict(complete=all(r['complete'] for r in results.values()), evaluation=results)
    planner = RolloutPolicy(seed=ctx.seed, candidates=ctx.config['t1']['candidates'], branches=ctx.config['t1']['branches'], b1=frozen.get('frozen_b1'))
    result = ctx.evaluate(planner, 'T1_rollout')
    return dict(complete=result['complete'], evaluation=result)
