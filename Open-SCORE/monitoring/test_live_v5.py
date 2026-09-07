"""Reporting correctness: real data separation and concurrent log writes."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('live_v5',Path(__file__).with_name('live_v5.py'))
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def row(family, won, red=8, blue=4, **extra):
    return dict(family_id=family,success_native=won,red_count=red,blue_count=blue,**extra)


def dataset(tmp):
    write(tmp/'shared/config_resolved.json',dict(seed=20260907,cells=[[8,4],[8,8]],
        eval_per_cell=1,validation_per_cell=1,t4=dict(rounds=1),smoke=False))
    return tmp


def log(run, task, method, split, ck, rows):
    p=run/task/'evaluations'/method/split/ck/'episodes.jsonl'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(dict(r,method_id=method,split=split,checkpoint_id=ck))+'\n' for r in rows),encoding='utf-8')
    return p


def test_partial_line_is_delayed_and_duplicate_families_not_counted(tmp_path):
    p=tmp_path/'records.jsonl'
    a=json.dumps(row('a',True)).encode()
    p.write_bytes(a+b'\n'+b'{"family_id":"b",')
    inputs=live.Inputs()
    assert len(inputs.rows(p))==1
    with p.open('ab') as f: f.write(b'"success_native":false}\n'+a+b'\n')
    assert len(inputs.rows(p))==3
    assert len(live.valid_episodes(inputs.rows(p)))==2
    p.write_bytes(a+b'\n')
    assert len(inputs.rows(p))==1


def test_errors_and_missing_success_are_not_losses():
    rows=[row('good',False),dict(family_id='missing'),row('error',False,execution_status='failed_execution')]
    assert [r['family_id'] for r in live.valid_episodes(rows)]==['good']


def test_completion_requires_each_cell_and_pending_is_not_zero():
    assert live.aggregate([],[[8,4],[8,8]],1)['rate'] is None
    assert not live.aggregate([row('a',True),row('b',True)],[[8,4],[8,8]],1)['complete']
    a=live.aggregate([row('a',True),row('b',False,blue=8)],[[8,4],[8,8]],1)
    assert a['complete'] and a['rate']==.5
    assert a['difficulty'][1]['rate'] is None


def test_pairing_uses_family_and_zero_difference_interval_is_not_zero():
    a=[row('a',True),row('b',False),row('unmatched',True)]
    b=[row('a',False),row('b',False),row('other',True)]
    p=live.paired(a,b)
    assert p['n']==2 and p['difference']==.5
    tied=live.paired([row('x',False)],[row('x',False)])
    assert tied['interval'][0]<0<tied['interval'][1]


def test_final_and_validation_are_distinct_and_not_best_checkpoint(tmp_path):
    run=dataset(tmp_path)
    log(run,'T3','t3_candidate','test','latest',[row('a',False),row('b',False,blue=8)])
    log(run,'T3','t3_candidate','test','best',[row('a',True),row('b',True,blue=8)])
    log(run,'T3','t3_candidate','validation','step_10',[row('a',True),row('b',False,blue=8)])
    log(run,'T3','t3_candidate','validation','step_100',[row('a',False)])
    s=live.collect(run,live.Inputs())
    m=s['methods']['t3_candidate']
    assert m['test']['rate']==0 and m['test']['complete']
    assert m['validation'][0]['rate']==.5
    assert not m['validation'][1]['complete']
    assert 'step_10' in live.validation_last(m)
    assert s['methods']['t3_autoregressive']['test']['rate'] is None
    assert not s['complete']


def test_teacher_comparison_is_same_subset_and_offline_mse_is_not_win_rate(tmp_path):
    run=dataset(tmp_path)
    log(run,'T4','t4_exit','validation','round_1',[row('a',False),row('b',True,blue=8)])
    log(run,'T4','t4_teacher_round_1','validation','round_1',[row('a',True)])
    p=run/'T6/training.jsonl';p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(kind='adv',epoch=40,loss=.01,validation=dict(advantage_mse=.02,brier=None)))+'\n')
    s=live.collect(run,live.Inputs())
    assert s['teacher_pairs']==[dict(round=1,n=1,teacher=1,student=0)]
    m=s['methods']['t6_adv']
    assert m['test']['rate'] is None and not m['validation']
    assert m['offline'][0]['advantage_mse']==.02


def test_html_escapes_external_log_strings_and_embeds_refresh(tmp_path):
    p=live.Page('test',tmp_path,tmp_path,dict(updated='2026-09-07 12:00:00'))
    p.paragraph('<script>evil()</script>')
    p.table(['name'],[['<img src=x>']])
    p.save()
    text=(tmp_path/'index.html').read_text(encoding='utf-8')
    assert '&lt;script&gt;evil()&lt;/script&gt;' in text
    assert '<script>evil()' not in text
    assert "cache:'no-store'" in text and '10000' in text


def test_plots_keep_training_validation_and_test_axes_separate(tmp_path):
    m=dict(id='sample',training=[dict(physical_steps=100,loss=.4,success_window=.5)],
        validation=[dict(complete=True,axis='Physical steps',x=100,rate=.25,difficulty=[])],
        offline=[],test_rows=[row('a',True),row('b',False)])
    series=live.build_plots(m,tmp_path,{})
    assert len({s['panel'] for s in series})==4
    assert (tmp_path/'curves.png').exists()
    assert next(s for s in series if 'Validation' in s['panel'])['points']==[(100,.25)]


def test_gate_validation_uses_frozen_selection_and_retains_source(tmp_path):
    run=dataset(tmp_path)
    log(run,'T6','gate0','validation','final_epoch',[row('a',True),row('b',True,blue=8)])
    selected=log(run,'T6','gate2','validation','final_epoch',[row('a',False),row('b',True,blue=8)])
    write(run/'T6/gate_selection.json',dict(chosen_threshold=.02,thresholds=[
        dict(threshold=0.,summary=dict(method_id='gate0')),
        dict(threshold=.02,summary=dict(method_id='gate2'))]))
    v=live.collect(run,live.Inputs())['methods']['t6_adv_gated']['validation']
    assert len(v)==1 and v[0]['rate']==.5 and v[0]['complete']
    assert v[0]['source']==selected.relative_to(run).as_posix()


def test_active_report_service_is_not_replaced(tmp_path):
    import os
    import psutil
    lock=tmp_path/'service.lock'
    token=dict(pid=os.getpid(),created=psutil.Process().create_time(),url='local-test')
    write(lock,token)
    before=lock.read_bytes()
    assert live.acquire_service(lock,dict(pid=-1,created=0))==token
    assert lock.read_bytes()==before


def test_full_report_set_and_final_label_wait_for_all_arms(tmp_path):
    run=dataset(tmp_path)
    s=live.collect(run,live.Inputs())
    live.render(run,s,{})
    root=run/'live'
    assert len(list((root/'tasks').glob('*/report.md')))==6
    assert len(list((root/'methods').glob('*/report.md')))==14
    assert '最终汇总' not in (root/'report.md').read_text(encoding='utf-8').splitlines()[0]
    assert '尚无数据' in (root/'methods/t3_autoregressive/report.md').read_text(encoding='utf-8')


def test_first_test_episode_zero_or_one_win_does_not_break_errorbar(tmp_path):
    run=dataset(tmp_path)
    log(run,'T6','t6_adv_gated','test','final_epoch',[row('first',False)])
    log(run,'T1','T1_rollout','test','latest',[row('first',True)])
    s=live.collect(run,live.Inputs())
    live.comparison_plot(s,run/'live',{})
    assert (run/'live/comparison.png').exists()
    assert live.wilson(0,1)[0]==0
    assert live.wilson(1,1)[1]==1
