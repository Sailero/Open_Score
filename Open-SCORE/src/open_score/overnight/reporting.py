"""Training-only curves with explicit validation selection and auditable data."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _read_rows(path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may leave only the final record incomplete. Other
            # corruption is an error rather than silently invented curves.
            if index != len(lines)-1:
                raise
    return rows


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def training_curve_data(output: Path) -> dict:
    """Extract actual points; no interpolation, smoothing loss, or test data."""
    output = Path(output)
    episodes = _read_rows(output/'episodes.jsonl')
    training = _read_rows(output/'training.jsonl')
    validation = _read_rows(output/'validation_history.jsonl')
    data = dict(schema='overnight-training-curves-v1',
                native_success=[], validation=[], metrics={},
                episodes_read=len(episodes), updates_read=len(training),
                skipped_validation_records=0, best_validation=None,
                notes=['Success uses native terminal wins, not shaped reward.',
                       'Rolling success uses up to 100 episodes within each phase; early windows use available episodes.',
                       'Validation is for checkpoint selection; held-out test results are excluded.',
                       'The x axis uses physical steps or recorded cumulative updates, never lagging training_seconds.'])
    histories = {}
    for row in episodes:
        if not _number(row.get('physical_steps')) or 'success' not in row:
            continue
        phase = row.get('phase', 'training')
        if phase not in {'training', 'simulation_teacher'}:
            continue
        history = histories.setdefault(phase, [])
        history.append(float(bool(row['success'])))
        data['native_success'].append(dict(physical_steps=int(row['physical_steps']),
            episode=row.get('episode'), success=bool(row['success']), phase=phase,
            rolling_success_rate=float(np.mean(history[-100:])), rolling_count=min(len(history), 100)))
    metric_names = ('loss', 'actor_loss', 'value_loss', 'entropy', 'q_mean', 'target_mean', 'teacher_loss')
    for name in metric_names:
        points = []
        for row in training:
            if not _number(row.get('updates')) or not _number(row.get(name)):
                continue
            phase = row.get('phase', 'training')
            if name == 'teacher_loss' and not (phase == 'simulation_teacher'
                    or row.get('teacher_examples', 0) > 0 or abs(row[name]) > 0):
                continue
            points.append(dict(updates=int(row['updates']), value=float(row[name]), phase=phase,
                               physical_steps=row.get('physical_steps')))
        data['metrics'][name] = points
    for row in validation:
        summary = row.get('summary', {})
        if summary.get('split') != 'validation' or not summary.get('complete', False):
            data['skipped_validation_records'] += 1
            continue
        table = [item for item in summary.get('table', [])
                 if item.get('method') == 'current' and _number(item.get('success_rate'))]
        if not table or not _number(row.get('steps')):
            data['skipped_validation_records'] += 1
            continue
        data['validation'].append(dict(physical_steps=int(row['steps']),
            mean_success_rate=float(np.mean([item['success_rate'] for item in table])),
            per_scale=[dict(scale=item['scale'], success_rate=float(item['success_rate']),
                            episodes=item.get('episodes')) for item in table]))
    best_path = output/'best_validation.json'
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding='utf-8-sig'))
        if _number(best.get('steps')) and _number(best.get('score')):
            match = next((point for point in data['validation'] if point['physical_steps'] == int(best['steps'])
                          and math.isclose(point['mean_success_rate'], float(best['score']), abs_tol=1e-10)), None)
            data['best_validation'] = dict(physical_steps=int(best['steps']), score=float(best['score']),
                                           matched_logged_validation=match is not None)
    return data


def plot_training(output: Path) -> Path:
    """Write training_curves.png and exact plotted points in a JSON sidecar."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = training_curve_data(output)
    with (output/'training_curves.json').open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(f"Training diagnostics | {output.name}\nValidation selects checkpoints; held-out tests are separate", fontsize=14)
    colors = {'training': '#2463a0', 'simulation_teacher': '#b86b16'}

    def setup(axis, title, xlabel, ylabel, *, fraction=False):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=.2)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.ticklabel_format(axis='x', style='plain', useOffset=False)
        if fraction:
            axis.set_ylim(-.02, 1.02)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))

    def empty(axis, label='No data recorded'):
        axis.text(.5, .5, label, ha='center', va='center', transform=axis.transAxes,
                  color='#656565', fontsize=11)

    axis = axes[0, 0]
    setup(axis, 'Native success: rolling up to 100 episodes', 'Physical environment steps', 'Success rate', fraction=True)
    for phase, label in [('training', 'Learner rollouts'), ('simulation_teacher', 'Teacher deployments')]:
        points = [point for point in data['native_success'] if point['phase'] == phase]
        if points:
            axis.plot([point['physical_steps'] for point in points], [point['rolling_success_rate'] for point in points],
                      label=label, color=colors[phase], linewidth=1.5)
    if axis.lines:
        axis.legend(loc='best', fontsize=8)
    else:
        empty(axis, 'No completed training episodes')

    axis = axes[0, 1]
    setup(axis, 'Validation mean across scales (selection only)', 'Physical environment steps', 'Success rate', fraction=True)
    points = data['validation']
    if points:
        axis.plot([point['physical_steps'] for point in points], [point['mean_success_rate'] for point in points],
                  'o-', color='#32815a', label='Complete validation nodes', linewidth=1.5, markersize=4)
        best = data['best_validation']
        if best and best['matched_logged_validation']:
            axis.scatter([best['physical_steps']], [best['score']], marker='*', s=170, color='#bd6c0b',
                         edgecolors='white', linewidths=.6, zorder=5,
                         label=f"Selected best: {best['physical_steps']:,} steps")
        axis.legend(loc='best', fontsize=8)
    else:
        empty(axis, 'No complete validation nodes\nHeld-out test data are never plotted here')

    def metrics_panel(axis, title, names, ylabel):
        setup(axis, title, 'Recorded cumulative learning updates', ylabel)
        for name, label, color in names:
            points = data['metrics'][name]
            if points:
                axis.plot([point['updates'] for point in points], [point['value'] for point in points],
                          'o-', markersize=3, linewidth=1.2, label=label, color=color)
        if axis.lines:
            axis.legend(loc='best', fontsize=8)
        else:
            empty(axis)

    metrics_panel(axes[1, 0], 'Learning objective (not evaluation performance)',
                  [('loss', 'Total / TD loss', '#2463a0'), ('actor_loss', 'PPO actor loss', '#b86b16'),
                   ('value_loss', 'PPO value loss', '#32815a')], 'Loss')
    metrics_panel(axes[1, 1], 'PPO candidate entropy', [('entropy', 'Categorical entropy', '#7d4b96')], 'Entropy (nats)')
    metrics_panel(axes[2, 0], 'Double DQN candidate values',
                  [('q_mean', 'Selected online Q', '#2463a0'), ('target_mean', 'Bellman target', '#b86b16')], 'Value')
    metrics_panel(axes[2, 1], 'Teacher supervision / auxiliary loss',
                  [('teacher_loss', 'Teacher cross entropy', '#b86b16')], 'Teacher loss')
    destination = output/'training_curves.png'
    fig.savefig(destination, dpi=150, facecolor='white')
    plt.close(fig)
    return destination


__all__ = ['plot_training', 'training_curve_data']
