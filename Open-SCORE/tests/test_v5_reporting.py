import gzip
import json

import pytest

from open_score.research_v5 import reporting as report


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


def episode(family, red=8, blue=4, won=False, **kwargs):
    return dict(record_type='episode', family_id=family, red_count=red, blue_count=blue,
                success_native=won, real_environment_steps=50, planner_sim_steps=25, **kwargs)


def write_shard(directory, row, times):
    path=directory/'families'/f'{row["family_id"]}.jsonl.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')
        for i, value in enumerate(times):
            stream.write(json.dumps(dict(record_type='event', family_id=row['family_id'],
                event_id=f'{row["family_id"]}_{i}', selection_wall_ms=value))+'\n')


def test_incremental_shard_recovery_and_true_event_latency(tmp_path):
    directory=tmp_path/'evaluation'
    first=episode('a', method_id='rule', checkpoint_id='frozen', decision_time_p50_ms=1)
    second=episode('b', decision_time_p50_ms=101)
    write_shard(directory, first, [1])
    jsonl(directory/'episodes.jsonl', [first])
    cache={}
    before=report.latency(directory, report.episode_rows(directory), cache)
    assert before['decision_p50_ms']==1
    # The compressed episode commits before the episode-index append.
    write_shard(directory, second, [100,101,102])
    rows=report.episode_rows(directory)
    after=report.latency(directory, rows, cache)
    assert len(rows)==2 and after['decision_samples']==4
    assert after['decision_p50_ms']==100.5
    assert after['decision_p95_ms']==pytest.approx(101.85)
    assert after['episode_median_latency_mean_ms']==51
    assert report.latency(directory, rows, cache)==after
    assert len(cache)==2


def test_macro_difficulty_and_equal_count_preserve_denominators(tmp_path):
    config=dict(cells=[[8,4],[16,8],[8,8]], eval_per_cell=2, seed=20260907)
    rows=[episode('a',won=True),episode('b',red=16,blue=8),episode('c',red=16,blue=8),
          episode('d',blue=8,won=True),episode('e',blue=8,won=True)]
    found={('candidate','latest'):dict(rows=rows,directory=tmp_path,task_id='T3')}
    r=report.comparison_rows(found,config,{})[0]
    assert r['success_rate']==.6
    assert r['macro_success_rate']==pytest.approx(2/3)
    assert r['difficulties'][0]['macro_success_rate']==.5
    assert r['difficulties'][0]['episodes']==3
    assert r['equal_count_success_rate']==1
    assert r['equal_count_episodes']==2
    assert not r['complete'] and r['evidence_status']=='inconclusive'


def test_full_counts_with_unpaired_families_are_not_gain_evidence(tmp_path):
    config=dict(cells=[[8,4]],eval_per_cell=2,seed=1)
    found={('rule','frozen'):dict(rows=[episode('ref1'),episode('ref2')],directory=tmp_path,task_id='T1'),
           ('candidate','latest'):dict(rows=[episode('ref1'),episode('other')],directory=tmp_path,task_id='T3')}
    row=next(r for r in report.comparison_rows(found,config,{}) if r['method_id']=='candidate')
    assert row['complete'] and row['paired_vs_rule']['pairs']==1
    assert row['evidence_status']=='inconclusive'
    same=report.paired_interval([episode('a')],[episode('a')])
    assert report.effect(same,True,True,False)=='no_observed_gain'
    assert same['interval95'][0]<0<same['interval95'][1]
    assert report.effect(same,True,True,True)=='smoke_only'


def test_training_axes_keep_rounds_epochs_and_physical_steps_distinct(tmp_path):
    t3=tmp_path/'T3'
    jsonl(t3/'training.jsonl',[
        dict(method_id='t3_candidate',phase='rule_bc',epoch=1,loss=.4),
        dict(method_id='t3_candidate',phase='train',physical_steps=1024,actor_loss=.3),
        dict(method_id='t3_autoregressive',phase='train',physical_steps=1024,actor_loss=.5)])
    jsonl(t3/'validation.jsonl',[dict(method_id='t3_candidate',physical_steps=1024,success_rate=.25)])
    series=report.curve_series(t3)
    bc=next(s for s in series if s['phase']=='rule_bc')
    assert bc['x_axis']=='Epoch' and bc['points']==[[1.,.4]]
    trained=[s for s in series if s['metric']=='actor_loss']
    assert len(trained)==2 and all(s['x_axis']=='Physical training steps' for s in trained)
    t4=tmp_path/'T4'
    jsonl(t4/'training.jsonl',[dict(method_id='t4_exit',phase='distill',round=r,epoch=1,loss=1/r) for r in (1,2)])
    assert {s['phase'] for s in report.curve_series(t4)}=={'distill_round_1','distill_round_2'}
    t5=tmp_path/'T5'
    jsonl(t5/'validation.jsonl',[dict(construction_episode=100,success_rate=.3)])
    report.atomic_json(t5/'evaluations/t5_bridge_grouping/validation/construction_200/summary.json',
                       dict(method_id='t5_bridge_grouping',checkpoint_id='construction_200',success_rate=.4))
    validation=report.curve_series(t5)
    assert len(validation)==1 and validation[0]['x_axis']=='Construction episodes'
    assert validation[0]['points']==[[100.,.3],[200.,.4]]
    t6=tmp_path/'T6'
    duplicated=[dict(kind='bce',epoch=1,loss=.5,validation=dict(brier=.2,ranking_accuracy=.6))]
    jsonl(t6/'training.jsonl',duplicated)
    jsonl(t6/'bce/training.jsonl',duplicated)
    assert len(report.curve_series(t6))==3
    assert all(len(s['points'])==1 for s in report.curve_series(t6))


def test_budget_records_keep_matched_work_separate_from_iterations(tmp_path):
    task=tmp_path/'T2'
    rows=[dict(state_id='a',family_id='f',depth=1,iterations=i,paired_gain_vs_rule=.1,
               simulation_physical_steps=50,decision_seconds=.2,iterations_executed=2)
          for i in (2,'matched_physical_work')]
    jsonl(task/'budget_curve.jsonl',rows+[rows[0]])
    parsed=report.budget_rows(task)
    assert len(parsed)==2 and {r['budget'] for r in parsed}=={2,'matched_physical_work'}
    assert all(r['simulation_physical_steps']==50 for r in parsed)


def test_summarize_waits_for_all_arms_tasks_and_serial_latency(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'plot_task',lambda path:None)
    run=tmp_path/'run'
    report.atomic_json(run/'shared/config_resolved.json',dict(cells=[[8,4]],eval_per_cell=1,smoke=False,seed=20260907))
    for task,methods in report.TASK_METHODS.items():
        report.atomic_json(run/task/'task_result.json',dict(complete=True,result={}))
        for method in methods:
            checkpoint=report.MAIN[method]
            directory=run/task/'evaluations'/method/'test'/checkpoint
            # With only committed shards, reporting must still find the method.
            write_shard(directory,episode('paired',won=method!='rule',method_id=method,checkpoint_id=checkpoint),[5,7])
    first=report.summarize(run)
    assert len(first['methods'])==14 and first['evaluation_complete'] and first['task_execution_complete']
    assert not first['complete'] and not first['serial_latency']['complete']
    serial=dict(complete=True,protocol='serial test protocol',methods=[dict(method_id=m,complete=True,
        mean_seconds=.02,p50_seconds=.01,p95_seconds=.04,rows=[dict(state_id='shared')]) for m in report.MAIN])
    report.atomic_json(run/'reports/serial_latency.json',serial)
    extra=run/'T3/evaluations/t3_candidate/test/best'
    write_shard(extra,episode('paired',won=True,method_id='t3_candidate',checkpoint_id='best'),[1])
    result=report.summarize(run)
    assert result['complete'] and len(result['supplementary_methods'])==1
    assert all(r['serial_decision_p50_ms']==10 and r['decision_p50_ms']==6 for r in result['methods'])
    assert (run/'comparison.csv').read_bytes().startswith(b'\xef\xbb\xbf')
    assert 'serial_latency.json' in (run/'final_report.md').read_text(encoding='utf-8')
    for name in ('comparison.json','diagnostics.csv','reproduction_status.csv','comparison_difficulty.csv','costs.csv'):
        assert (run/name).exists()


def test_pinned_references_are_reported_without_self_referential_mapping(tmp_path):
    report.atomic_json(tmp_path/'T3/source_references.json',dict(task_id='T3',references=dict(ppo=dict(url='paper')),
        original_environments_run=False,code_vendored=False))
    (tmp_path/'reproduction_status.csv').write_text('generated',encoding='utf-8')
    task=dict(task_id='T3',execution_status='running',phase='train',result={})
    row=report.reproduction_rows(tmp_path,[task],[])[0]
    assert row['source_mapping_recorded'] and row['source_reference']==dict(ppo=dict(url='paper'))
    assert [p.replace('\\','/') for p in row['mapping_files']]==['T3/source_references.json']
    assert row['original_environments_run'] is False


def test_ledger_write_is_scoped_and_preserves_user_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path/'Open-SCORE')
    run=tmp_path/'Open-SCORE/outputs/v5_parallel/v5_20260907_main'
    report.atomic_json(run/'shared/config_resolved.json',dict(smoke=False,seed=20260907))
    ledger=tmp_path/'实验记录.md'
    prefix='用户原文一\r\n<!-- V5_CURRENT_START -->'
    suffix='<!-- V5_CURRENT_END -->\r\n用户追加二\r\n'
    original=(prefix+'\r\n旧阶段\r\n'+suffix).encode('utf-8')
    ledger.write_bytes(original)
    output=dict(updated='2026-09-07',complete=False,methods=[],tasks=[dict(task_id='T3',execution_status='running',phase='PPO')])
    assert report.update_ledger(run,output)['updated']
    changed=ledger.read_bytes()
    assert changed.startswith(prefix.encode('utf-8')) and changed.endswith(suffix.encode('utf-8'))
    assert '尚未全部完成' in changed.decode('utf-8')
    assert '当前进度与结果报告' in changed.decode('utf-8')
    report.atomic_json(run/'shared/config_resolved.json',dict(smoke=True))
    assert not report.update_ledger(run,output)['updated'] and ledger.read_bytes()==changed
    assert not report.update_ledger(run.parent/'another_run',output)['updated']


def test_independent_score_audit_fields_survive_reporting(tmp_path):
    import torch
    path=tmp_path/'T6/diagnostics/t6_adv/diagnostic/a.pt'
    path.parent.mkdir(parents=True)
    summary=dict(method_id='t6_adv',state_id='a',state_set='diagnostic',non_tie_pairs=3,
                 ranking_accuracy=2/3,paired_advantage_mse=.1,probability_brier=.2,probability_ece=.05)
    torch.save(dict(summary=summary,predictions=[.1,.2]),path)
    row=report.diagnostic_rows(tmp_path,{})[0]
    assert row['ranking_accuracy']==2/3 and row['probability_ece']==.05
    assert row['source'].replace('\\','/')=='T6/diagnostics/t6_adv/diagnostic/a.pt'


def test_generated_task_summaries_do_not_complete_workers_or_replace_method_analysis(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'plot_task',lambda path:None)
    report.atomic_json(tmp_path/'shared/config_resolved.json',dict(cells=[[8,4]],eval_per_cell=1,smoke=True,seed=1))
    for name in ('T3','T4','T5'):
        report.atomic_json(tmp_path/name/'progress.json',dict(phase='training'))
    (tmp_path/'T5/analysis.md').write_text('existing method analysis',encoding='utf-8')
    first=report.summarize(tmp_path)
    assert len(first['task_analysis'])==3
    assert report.read_json(tmp_path/'T3/summary.json')['status']=='running'
    assert '尚无末期测试或诊断记录' in (tmp_path/'T4/analysis.md').read_text(encoding='utf-8')
    assert (tmp_path/'T5/analysis.md').read_text(encoding='utf-8')=='existing method analysis'
    second=report.summarize(tmp_path)
    assert not second['complete']
    generated=report.read_json(tmp_path/'T3/summary.json')
    assert generated['worker_result']=={}  # no recursive generated-summary nesting
    report.atomic_json(tmp_path/'T3/task_result.json',dict(complete=True,result=dict(complete=True,arms=[])))
    report.summarize(tmp_path)
    assert report.read_json(tmp_path/'T3/summary.json')['status']=='complete'
    assert 'task_analysis.md' in (tmp_path/'final_report.md').read_text(encoding='utf-8')


def test_t1_t2_diagnostics_merge_actual_episode_spec_keys(tmp_path):
    from open_score.research_v5.protocol import episode_spec
    spec=episode_spec(20260907,0,'diagnostic').to_dict()
    for task in ('T1','T2'):
        report.atomic_json(tmp_path/task/'own_diagnostics/own.json',dict(method=task,state_id='own',
            family_id=spec['family_id'],episode_spec=spec,paired_gain_vs_rule=.125,simulation_physical_steps=100))
    report.atomic_json(tmp_path/'T1/common_diagnostics/common.json',dict(state_id='common',
        family_id=spec['family_id'],episode_spec=spec,paired_gain_vs_rule=dict(rule=0.,singleton=.25),
        selection_all_zero=False,selection_all_tie=False,
        group_only=dict(singleton=dict(mean_difference=.125,branch_vector_changed=True))))
    rows=report.diagnostic_rows(tmp_path,{})
    assert len(rows)==5
    assert all(r['family_id']==spec['family_id'] and r['red_count']==spec['red_count'] and
               r['blue_count']==spec['blue_count'] for r in rows)
    assert {r['state_set'] for r in rows}=={'onpolicy_diagnostic','diagnostic','same_assignment_grouping'}
    assert next(r for r in rows if r['task_id']=='T2')['selected_vs_rule_verification_gain']==.125
