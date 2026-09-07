"""One entry point for compact experiments: run, resume, status and report."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .protocol import ROOT, defaults, episode_spec
from .storage import Store, method_dir

JOBS = {
    'rollout': ('T1', 't1_rollout', [('T1_rollout', 'latest')]),
    'mcts_dpw': ('T2', 't2_mcts', [('T2_mcts_dpw', 'latest')]),
    'candidate_ppo': ('T3', 't3_ppo', [('t3_candidate', 'latest')]),
    'ar_ppo': ('T3', 't3_ppo', [('t3_autoregressive', 'latest')]),
    'exit': ('T4', 't4_exit', [('t4_exit', 'latest')]),
    'bridge': ('T5', 't5_bridge_grouping', [('t5_bridge_grouping', 'final_construction')]),
    'paired_value': ('T6', 't6_paired_value', [('t6_bce','final_epoch'),('t6_adv','final_epoch'),('t6_adv_gated','final_epoch')]),
    'baselines': ('T1', 't1_rollout', [(m,'frozen') for m in ('rule','grand','singleton','frozen_b1','frozen_b3')]),
}

def now():
    return datetime.now(timezone.utc).isoformat()

def configure_threads():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    import torch
    torch.set_num_threads(1)

def configuration(path):
    return json.loads((Path(path)/'config.json').read_text(encoding='utf-8'))

def done(path, method, config):
    store=Store(method_dir(path,method))
    expected=int(config['eval_per_cell'])*len(config['cells'])
    try:
        return all(len(store.episodes(arm,'test',checkpoint))==expected for arm,checkpoint in JOBS[method][2])
    finally:
        store.close()

def prepare(path, args):
    from .runtime import TaskContext
    from .simulator import make_env
    from open_score.research_v4.actions import rule_grouping
    from open_score.grouping.storage import atomic_json
    path.mkdir(parents=True,exist_ok=True)
    if not (path/'config.json').exists():
        if args.command=='resume': raise FileNotFoundError('No experiment to resume')
        config=defaults(multiplier=args.multiplier,seed=args.seed)
        if args.config:
            import yaml
            def merge(target, source):
                for key,value in source.items():
                    if isinstance(value,dict) and isinstance(target.get(key),dict): merge(target[key],value)
                    else: target[key]=value
            merge(config,yaml.safe_load(Path(args.config).read_text(encoding='utf-8')))
        config['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        config['remaining_diagnostics']='cancelled_by_user'
        retained=ROOT/'outputs/v5/config.json'
        if retained.exists(): config['frozen_assets']=json.loads(retained.read_text(encoding='utf-8')).get('frozen_assets',{})
        if not config.get('frozen_assets'):
            raise ValueError('Set retained frozen_assets paths in the configuration')
        atomic_json(path/'config.json',config)
    config=configuration(path)
    shared=Store(path/'shared')
    for split,name,n in [('test','evaluation',config['eval_per_cell']),('validation','validation',config['validation_per_cell'])]:
        if shared.get(name+'_manifest.json') is None:
            shared.put(name+'_manifest.json',dict(split=split,episodes=[episode_spec(config['seed'],i,split,'shared',config['cells']).to_dict() for i in range(n*len(config['cells']))]))
    if shared.load_torch('bc_records.pt') is None:
        records=[]; physical=0
        for i in range(config['bc_episodes']):
            key=f'bc_families/{i:05d}.pt'; data=shared.load_torch(key)
            if data is None:
                spec=episode_spec(config['seed'],i,'train_bc','shared',config['cells'])
                env=make_env(spec); rows=[]
                try:
                    while not env.done:
                        state=env.state(); action=rule_grouping(state)
                        rows.append(dict(state=state,action=action,family_id=spec.family_id,episode_spec=spec.to_dict()))
                        env.step(action)
                    data=dict(rows=rows,physical_steps=env.state().step)
                finally: env.close()
                shared.save_torch(key,data)
            records.extend(data['rows']); physical+=data['physical_steps']
            if i%10==0: print(f'[prepare] rule initialization {i+1}/{config["bc_episodes"]}',flush=True)
        shared.save_torch('bc_records.pt',dict(rows=records,physical_steps=physical,episodes=config['bc_episodes']))
    return config

def execute(path,method):
    from .runtime import TaskContext
    task,module,_=JOBS[method]
    config=configuration(path)
    if done(path,method,config): return
    config=dict(config,selected_method=method)
    ctx=TaskContext(path,task,config);ctx.select_method(method)
    ctx.progress('starting',method=method)
    try:
        result=importlib.import_module(f'.tasks.{module}',__package__).run(ctx)
        ctx.store.put('result',result)
        ctx.progress('complete',complete=True,method=method)
    except Exception as error:
        ctx.store.put('error',dict(time=now(),error=str(error),traceback=traceback.format_exc()))
        ctx.progress('error',error=str(error),method=method)
        raise

def running_pid(path):
    import psutil
    with Store(path/'shared') as store:
        value=store.get('scheduler',{})
    pid=value.get('pid')
    try:
        process=psutil.Process(pid)
        command=' '.join(process.cmdline())
        return pid if str(path) in command and 'research_v5' in command and process.is_running() else None
    except (psutil.Error,TypeError): return None

def status(path,watch=False):
    config=configuration(path)
    while True:
        print(f'[{datetime.now():%H:%M:%S}] '+str(path),flush=True)
        for method in JOBS:
            store=Store(method_dir(path,method)); progress=store.get('progress',{})
            finished=done(path,method,config)
            extra=''
            if not finished:
                for key,total in [('physical_steps','total_physical_steps'),('completed_episodes','total_episodes'),('epoch','epochs')]:
                    if key in progress: extra+=f' {key}={progress[key]}/{progress.get(total,"?")}'
                if progress.get('eta_seconds') is not None: extra+=f' phase ETA {progress["eta_seconds"]/60:.1f} min'
                if progress.get('recent_success') is not None: extra+=f' train win {progress["recent_success"]:.1%}'
            rows=[r for arm,ck in JOBS[method][2] for r in store.episodes(arm,'test',ck)]
            store.close()
            if rows: extra+=f' final test {sum(bool(r["success_native"]) for r in rows)}/{len(rows)}'
            print(f'  {method}: {"complete" if finished else progress.get("phase","pending")}{extra}',flush=True)
        if not watch or all(done(path,m,config) for m in JOBS): return
        time.sleep(10)

def run(path,args):
    from .reporting import summarize
    config=prepare(path,args)
    if running_pid(path):
        print('Existing experiment is running; attaching progress only.',flush=True)
        status(path,True);return
    shared=Store(path/'shared')
    methods=[args.method] if args.method else list(JOBS)
    queue=[m for m in methods if not done(path,m,config)]
    active={};streams={};last_display=last_report=0.
    shared.put('scheduler',dict(pid=os.getpid(),started=now(),methods=methods))
    try:
        while queue or active:
            for method in list(queue):
                if len(active)>=args.workers: break
                # The two PPO arms share one task allocation and execute sequentially.
                if method in ('candidate_ppo','ar_ppo') and any(m in active for m in ('candidate_ppo','ar_ppo')): continue
                queue.remove(method)
                folder=method_dir(path,method);folder.mkdir(parents=True,exist_ok=True)
                streams[method]=(folder/'run.log').open('a',encoding='utf-8')
                command=[sys.executable,'-X','utf8','-u','-m','open_score.research_v5.orchestrate','resume','--run-dir',str(path),'--method',method,'--worker']
                active[method]=subprocess.Popen(command,cwd=ROOT,stdout=streams[method],stderr=subprocess.STDOUT)
            for method,process in list(active.items()):
                if process.poll() is not None:
                    code=process.returncode;streams.pop(method).close();del active[method]
                    if code: raise RuntimeError(f'{method} stopped ({code}); see {method}/run.log')
            current=time.monotonic()
            if current-last_display>=10: status(path);last_display=current
            if current-last_report>=60:
                try: summarize(path)
                except Exception as error: print(f'Report update: {error}',flush=True)
                last_report=current
            if active: time.sleep(.5)
        summary=summarize(path)
        shared.put('run_result',dict(complete=summary['complete'],finished=now(),cancelled='remaining_optional_diagnostics'))
    finally:
        if active:
            import psutil
            for process in active.values():
                try:
                    children=psutil.Process(process.pid).children(recursive=True)
                    for child in children: child.terminate()
                    process.terminate()
                except psutil.Error: pass
            for stream in streams.values(): stream.close()
        shared.put('scheduler',dict(pid=None,stopped=now()))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','resume','status','report'])
    parser.add_argument('--run-dir',default='outputs/v5')
    parser.add_argument('--method',choices=list(JOBS))
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--config')
    parser.add_argument('--seed',type=int,default=20260907)
    parser.add_argument('--multiplier',type=float,default=.5)
    args=parser.parse_args(argv);path=Path(args.run_dir).resolve();configure_threads()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.command=='status': status(path,args.watch)
    elif args.command=='report':
        from .reporting import summarize
        summarize(path)
    elif args.worker: execute(path,args.method)
    else: run(path,args)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
