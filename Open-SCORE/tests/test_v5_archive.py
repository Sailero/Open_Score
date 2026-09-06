from pathlib import Path
import subprocess

from open_score.research_v4.runner import atomic_json


def test_result_commit_excludes_unrelated_staged_user_changes(tmp_path,monkeypatch):
    import open_score.research_v5.archive as archival
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=tmp_path,text=True).strip()
    git('init','-q');git('config','user.email','test@example.invalid');git('config','user.name','Test')
    (tmp_path/'user.txt').write_text('old',encoding='utf-8')
    git('add','user.txt');git('commit','-qm','initial')
    (tmp_path/'user.txt').write_text('user staged work',encoding='utf-8');git('add','user.txt')
    project=tmp_path/'project';run=project/'outputs/run'
    (run/'reports').mkdir(parents=True)
    (run/'final_report.md').write_text('completed evidence',encoding='utf-8')
    atomic_json(run/'run_result.json',{'complete':True})
    monkeypatch.setattr(archival,'ROOT',project)
    result=archival.commit_results(run,{'smoke':False,'seed':20260907})
    assert result['status']=='committed'
    assert git('show','HEAD:user.txt')=='old'
    assert git('diff','--cached','--name-only')=='user.txt'
    assert git('show','HEAD:project/outputs/run/final_report.md')=='completed evidence'
