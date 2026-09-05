"""Plots preserve measured data and keep test metrics out of model selection."""
import json

import pytest

from open_score.overnight.reporting import plot_training, training_curve_data


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


def test_plot_preserves_actual_points_phases_and_validation_selection(tmp_path):
    write_rows(tmp_path/'episodes.jsonl', [
        dict(episode=1, physical_steps=10, success=True, phase='simulation_teacher'),
        dict(episode=2, physical_steps=20, success=False, phase='training'),
        dict(episode=3, physical_steps=40, success=True, phase='training'),
        dict(episode=4, physical_steps=70, success=True, phase='training')])
    write_rows(tmp_path/'training.jsonl', [
        dict(updates=1, physical_steps=10, phase='simulation_teacher', teacher_loss=1.2, teacher_examples=8),
        dict(updates=3, physical_steps=40, loss=.25, entropy=1.3, actor_loss=-.2, value_loss=.9, teacher_loss=.5,
             optimizer_steps=19, optimizer_steps_this_update=8, training_seconds=0),
        dict(updates=8, physical_steps=70, loss=.125, entropy=.7, teacher_loss=.2,
             optimizer_steps=7, training_seconds=0)])
    def validation(steps, split, complete, score):
        return dict(steps=steps, summary=dict(split=split, complete=complete, table=[
            dict(method='current', scale=8, episodes=10, success_rate=score),
            dict(method='current', scale=16, episodes=10, success_rate=score+.2),
            dict(method='dynamic_rule', scale=8, episodes=10, success_rate=1.)]))
    write_rows(tmp_path/'validation_history.jsonl', [validation(40, 'validation', True, .2),
        validation(50, 'held_out_test', True, .8), validation(60, 'validation', False, .9),
        validation(70, 'validation', True, .4)])
    (tmp_path/'best_validation.json').write_text(json.dumps(dict(steps=70, score=.5)))
    artifact = plot_training(tmp_path)
    assert artifact == tmp_path/'training_curves.png'
    assert artifact.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert artifact.stat().st_size > 20_000
    plotted = json.loads((tmp_path/'training_curves.json').read_text())
    assert plotted['episodes_read'] == 4 and plotted['updates_read'] == 3
    assert [point['physical_steps'] for point in plotted['native_success']] == [10, 20, 40, 70]
    assert [point['rolling_success_rate'] for point in plotted['native_success']] == pytest.approx([1, 0, .5, 2/3])
    assert [point['updates'] for point in plotted['metrics']['loss']] == [3, 8]
    assert [point['value'] for point in plotted['metrics']['loss']] == [.25, .125]
    assert [point['value'] for point in plotted['metrics']['teacher_loss']] == [1.2, .5, .2]
    assert [point['physical_steps'] for point in plotted['validation']] == [40, 70]
    assert [point['mean_success_rate'] for point in plotted['validation']] == pytest.approx([.3, .5])
    assert plotted['skipped_validation_records'] == 2
    assert plotted['best_validation']['matched_logged_validation']


def test_empty_logs_produce_explicit_empty_data_artifact(tmp_path):
    artifact = plot_training(tmp_path)
    assert artifact.is_file()
    data = training_curve_data(tmp_path)
    assert data['native_success'] == data['validation'] == []
    assert all(not points for points in data['metrics'].values())
    assert data['best_validation'] is None


def test_rolling_success_uses_exactly_latest_100_and_q_metrics_survive(tmp_path):
    write_rows(tmp_path/'episodes.jsonl', [dict(episode=i+1, physical_steps=(i+1)*50, success=i == 0)
                                          for i in range(101)])
    write_rows(tmp_path/'training.jsonl', [dict(updates=10, loss=.8, q_mean=2., target_mean=2.5, teacher_loss=0.)])
    data = training_curve_data(tmp_path)
    assert data['native_success'][99]['rolling_success_rate'] == .01
    assert data['native_success'][100]['rolling_success_rate'] == 0.
    assert data['native_success'][100]['rolling_count'] == 100
    assert data['metrics']['q_mean'][0]['value'] == 2.
    assert data['metrics']['target_mean'][0]['value'] == 2.5
    assert data['metrics']['teacher_loss'] == []


def test_live_partial_final_line_is_ignored_but_middle_corruption_raises(tmp_path):
    path = tmp_path/'training.jsonl'
    path.write_text('{"updates": 1, "loss": 0.4}\n{"updates":', encoding='utf-8')
    assert training_curve_data(tmp_path)['metrics']['loss'][0]['value'] == .4
    path.write_text('{"updates":\n{"updates": 1, "loss": 0.4}\n', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        training_curve_data(tmp_path)
