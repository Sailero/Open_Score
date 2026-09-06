"""Fixed-quota six-task runner. Wall-clock estimates never stop formal work."""
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

from .protocol import ROOT, TASKS, VERSION, defaults, digest, episode_spec, source_identity

MODULES=dict(T1='t1_rollout',T2='t2_mcts',T3='t3_ppo',T4='t4_exit',T5='t5_bridge_grouping',T6='t6_paired_value')


def now():
    return datetime.now(timezone.utc).isoformat()


def configure_threads():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    import torch
    torch.set_num_threads(1)


def hardware_profile():
    import platform
    import torch
    import psutil
    profile=dict(timestamp_utc=now(),platform=platform.platform(),python=sys.version,
                 executable=sys.executable,torch=torch.__version__,cuda_runtime=torch.version.cuda,
                 cpu_logical=psutil.cpu_count(),cpu_physical=psutil.cpu_count(logical=False),
                 memory_total_bytes=psutil.virtual_memory().total,memory_available_bytes=psutil.virtual_memory().available,
                 cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        device=torch.cuda.get_device_properties(0)
        profile.update(gpu_name=device.name,gpu_memory_bytes=device.total_memory)
        model=torch.nn.Linear(16,1,device='cuda')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        loss=model(torch.ones(8,16,device='cuda')).square().mean()
        loss.backward(); optimizer.step(); torch.cuda.synchronize()
        profile['cuda_update_verified']=bool(torch.isfinite(loss))
        del model,optimizer,loss
        torch.cuda.empty_cache()
    try:
        profile['cpu_model']=subprocess.check_output(['powershell','-NoProfile','-Command',
            '(Get-CimInstance Win32_Processor).Name'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        profile['cpu_model']=platform.processor()
    return profile


def immutable_files(run_dir,config):
    from open_score.research_v4.runner import file_hash
    run_dir=Path(run_dir)
    shared=run_dir/'shared'
    paths=[shared/name for name in ('config_resolved.json','protocol_resolved.json','evaluation_manifest.json',
                                   'validation_manifest.json','split_manifest.json','bc_records.pt','bc_manifest.json')]
    paths.extend((shared/'datasets').rglob('family_*.pt'))
    assets=Path(config.get('frozen_v4_assets',ROOT/'outputs/v4_comparison/shared/seed_20260906'))
    paths.extend([assets/'count_model.json',assets/'global_model/best.pt'])
    return {str(p.resolve()):file_hash(p) if p.exists() else None for p in paths}


def validate_frozen(run_dir):
    from open_score.research_v4.runner import read_json,file_hash
    path=Path(run_dir)
    frozen=read_json(path/'shared/budget_manifest.json',{})
    if not frozen.get('frozen'):
        raise ValueError('Freeze the experiment before starting a task')
    config=read_json(path/'shared/config_resolved.json')
    if frozen['source_hash']!=source_identity()['source_hash'] or frozen['protocol_hash']!=digest(config):
        raise ValueError('Task source/config differs from the frozen experiment')
    for name,expected in frozen.get('immutable_files',{}).items():
        p=Path(name)
        actual=file_hash(p) if p.exists() else None
        if actual!=expected:
            raise ValueError(f'Frozen data, evaluation manifest or baseline asset changed: {p}')
    return frozen


def prepare(run_dir,config, *, freeze=False):
    from open_score.research_v4.runner import atomic_json,read_json
    from open_score.grouping.storage import atomic_checkpoint
    from .runtime import TaskContext
    from .simulator import make_env
    from open_score.research_v4.actions import rule_grouping
    run_dir=Path(run_dir).resolve()
    shared=run_dir/'shared'
    shared.mkdir(parents=True,exist_ok=True)
    existing=read_json(shared/'config_resolved.json')
    frozen=read_json(shared/'budget_manifest.json',{})
    if freeze and not config.get('smoke') and not read_json(run_dir/'calibration_report.json',{}).get('usable'):
        raise ValueError('A completed usable production calibration is required before formal freeze')
    if frozen.get('frozen'):
        if existing != config or frozen['source_hash'] != source_identity()['source_hash']:
            raise ValueError('Frozen source/configuration cannot be replaced; use a new run directory')
        validate_frozen(run_dir)
        return run_dir
    if existing and existing != config:
        if read_json(shared/'budget_manifest.json',{}).get('frozen'):
            raise ValueError('Frozen configuration differs; use a new run directory')
    atomic_json(shared/'config_resolved.json',config)
    identity=source_identity()
    atomic_json(shared/'protocol_resolved.json',dict(version=VERSION,protocol_hash=digest(config),**identity))
    if not (shared/'hardware_profile.json').exists():
        atomic_json(shared/'hardware_profile.json',hardware_profile())
    for split,name,n in [('test','evaluation',config['eval_per_cell']),('validation','validation',config['validation_per_cell'])]:
        specs=[episode_spec(config['seed'],i,split,'shared',config['cells']).to_dict()
               for i in range(n*len(config['cells']))]
        atomic_json(shared/f'{name}_manifest.json',dict(split=split,episodes=specs,hash=digest(specs)))
    atomic_json(shared/'split_manifest.json',dict(version=VERSION,
        isolation='family ID includes seed, namespace, split, red/blue counts and opening replicate',
        namespaces=['shared','T1','T2','T3','T4:round','T5','T6'],
        excluded_preflight_opening_seeds=list(range(935000,935020)),
        test_reuses_v4=False))
    ctx=TaskContext(run_dir,'shared',config)
    bc_path=shared/'bc_records.pt'
    if not bc_path.exists():
        import torch
        records=[]
        physical_steps=0
        bcdir=shared/'bc_families';bcdir.mkdir(exist_ok=True)
        for i in range(config['bc_episodes']):
            checkpoint=bcdir/f'{i:05d}.pt'
            if checkpoint.exists():
                data=torch.load(checkpoint,map_location='cpu',weights_only=False)
            else:
                spec=episode_spec(config['seed'],i,'train_bc','shared',config['cells'])
                env=make_env(spec);rows=[]
                try:
                    while not env.done:
                        state=env.state();action=rule_grouping(state)
                        rows.append(dict(state=state,action=action,family_id=spec.family_id,episode_spec=spec.to_dict()))
                        env.step(action)
                    data=dict(rows=rows,physical_steps=env.state().step)
                finally:
                    env.close()
                atomic_checkpoint(checkpoint,data)
            records.extend(data['rows']);physical_steps+=data['physical_steps']
            if i%10==0:
                print(f'[prepare] rule BC {i+1}/{config["bc_episodes"]} episodes',flush=True)
            ctx.progress('prepare_bc',completed_episodes=i+1,total_episodes=config['bc_episodes'])
        atomic_checkpoint(bc_path,dict(rows=records,physical_steps=physical_steps,episodes=config['bc_episodes'],
                                     source_hash=identity['source_hash']))
        atomic_json(shared/'bc_manifest.json',dict(episodes=config['bc_episodes'],records=len(records),
                    real_physical_steps=physical_steps,cost_charged_once=True,seed=config['seed']))
    ctx.collect_states(config['diagnostic_states'],'diagnostic',namespace='shared')
    if freeze:
        atomic_json(shared/'budget_manifest.json',dict(frozen=True,frozen_at=now(),config=config,
                    protocol_hash=digest(config),source_hash=identity['source_hash'],git_commit=identity['git_commit'],
                    immutable_files=immutable_files(run_dir,config),
                    wall_clock_stop=None,expected_test_executions=14*config['eval_per_cell']*len(config['cells'])))
    ctx.progress('prepared',complete=True)
    return run_dir


def execute_task(run_dir,task):
    from .runtime import TaskContext
    from open_score.research_v4.runner import atomic_json,directory_lock,read_json
    from open_score.grouping.storage import seed_everything
    from .reporting import plot_task
    validate_frozen(run_dir)
    ctx=TaskContext(run_dir,task)
    from .references import register
    register(ctx)
    seed_everything(ctx.seed)
    with directory_lock(ctx.output):
        ctx.progress('starting',execution_status='running')
        try:
            module=importlib.import_module(f'open_score.research_v5.tasks.{MODULES[task]}')
            started=time.monotonic()
            result=module.run(ctx)
            register(ctx,append_markdown=True)
            atomic_json(ctx.output/'task_result.json',dict(complete=True,task_id=task,finished=now(),
                        elapsed_seconds=time.monotonic()-started,result=result))
            ctx.progress('complete',execution_status='completed',complete=True)
            plot_task(ctx.output)
        except BaseException as error:
            atomic_json(ctx.output/'execution_error.json',dict(task_id=task,time=now(),error=str(error),traceback=traceback.format_exc()))
            ctx.progress('execution_error',execution_status='failed_execution',error=str(error),complete=False)
            raise


def resource_snapshot(active):
    import psutil
    rows=[]
    for task,process in active.items():
        try:
            parent=psutil.Process(process.pid)
            children=parent.children(recursive=True)
            processes=[parent,*children]
            rows.append(dict(task_id=task,pid=process.pid,rss_bytes=sum(p.memory_info().rss for p in processes if p.is_running()),
                             cpu_seconds=sum(sum(p.cpu_times()[:2]) for p in processes if p.is_running()),process_count=len(processes)))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    gpu=None
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                                        '--format=csv,noheader,nounits'],text=True,timeout=3).strip()
        values=output.splitlines()[0].split(',')
        gpu=dict(utilization_percent=float(values[0]),memory_used_mib=float(values[1]),memory_total_mib=float(values[2]))
    except (OSError,ValueError,subprocess.SubprocessError):
        pass
    return dict(time=now(),tasks=rows,gpu=gpu,available_memory_bytes=psutil.virtual_memory().available)


def display(run_dir):
    from open_score.research_v4.runner import read_json
    config=read_json(Path(run_dir)/'shared/config_resolved.json',{})
    lines=[f'[{datetime.now().strftime("%H:%M:%S")}] v5.1 | seed={config.get("seed")} | 15 mixed cells | fixed quotas']
    for task in TASKS:
        p=read_json(Path(run_dir)/task/'progress.json',{})
        phase=p.get('phase','queued')
        parts=[]
        for done,total,label in [('physical_steps','total_physical_steps','real'),('completed_episodes','total_episodes','episodes'),
                                 ('completed_states','total_states','states'),('epoch','epochs','epoch'),
                                 ('round','rounds','round'),('completed','total','progress')]:
            if done in p:
                parts.append(f'{label}={p[done]}/{p.get(total,"?")}')
        for key,label in [('method_id','arm'),('simulation_physical_steps','sim'),('optimizer_steps','updates'),
                          ('evaluation_success_rate','eval_win'),('last_validation_success_rate','last_val_win'),
                          ('rolling_success_rate','train_win'),('eta_seconds','stage_ETA_s'),
                          ('throughput_per_second','units/s'),('evaluation_real_steps','eval_real'),
                          ('deployment_rate','ep_deployed'),('current_member_deployment_rate','member_deployed')]:
            if p.get(key) is not None:
                value=p[key];parts.append(f'{label}={value:.3f}' if isinstance(value,float) else f'{label}={value}')
        if not p.get('method_id') and p.get('method'):
            parts.append('arm='+str(p['method']))
        if p.get('error'):
            parts.append('ERROR='+str(p['error'])[:100])
        if task=='T3' and p.get('phase')=='ppo':
            rows=tail_records(Path(run_dir)/task/'episodes.jsonl')
            method='t3_'+str(p.get('method',''))
            for ratio,label in [(.5,'easy'),(.75,'medium'),(1.,'equal')]:
                subset=[r for r in rows if r.get('method_id')==method and r.get('blue_count',0)/max(1,r.get('red_count',1))==ratio]
                if subset:
                    parts.append(f'{label}_train={sum(r["success_native"] for r in subset)}/{len(subset)}')
        lines.append(f'{task:2s} {phase:20s} '+ ' | '.join(parts))
    print('\n'.join(lines),flush=True)


def tail_records(path, byte_limit=262144):
    """Bounded recent native training outcomes; not an evaluation estimator."""
    if not path.exists():
        return []
    with path.open('rb') as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-byte_limit))
        data=stream.read().splitlines()
    if size>byte_limit:
        data=data[1:]
    rows=[]
    for line in data[-300:]:
        try:
            rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):
            pass
    return rows


def run_all(run_dir,parallel=6,tasks=TASKS,allow_unfrozen=False):
    from open_score.research_v4.runner import read_json,atomic_json,directory_lock
    from .runtime import append
    from .reporting import summarize
    run_dir=Path(run_dir).resolve()
    frozen=read_json(run_dir/'shared/budget_manifest.json',{})
    if not frozen.get('frozen') and not allow_unfrozen:
        raise ValueError('Run calibrate/freeze before formal training')
    if frozen.get('frozen'):
        validate_frozen(run_dir)
    logs=run_dir/'logs';logs.mkdir(exist_ok=True)
    def refresh_report():
        try:
            summary=summarize(run_dir)
            atomic_json(run_dir/'report_status.json',dict(updated=now(),status='ok',complete=summary['complete']))
            return summary
        except Exception as error:
            atomic_json(run_dir/'report_status.json',dict(updated=now(),status='error',error=str(error),
                        traceback=traceback.format_exc(),note='Task workers continue; reporting can be retried independently.'))
            print(f'[report error; workers continue] {error}',flush=True)
            return None
    with directory_lock(run_dir):
        queue=[t for t in tasks if not read_json(run_dir/t/'task_result.json',{}).get('complete')]
        active={};streams={};finished=[];started=time.monotonic();started_at=now();last_print=last_resource=0.;last_report=started
        try:
            while queue or active:
                while queue and len(active)<parallel:
                    task=queue.pop(0)
                    streams[task]=(logs/f'{task}.log').open('a',encoding='utf-8')
                    command=[sys.executable,'-u','-m','open_score.research_v5.orchestrate','task','--run-dir',str(run_dir),'--task',task]
                    active[task]=subprocess.Popen(command,cwd=ROOT,stdout=streams[task],stderr=subprocess.STDOUT)
                for task,process in list(active.items()):
                    code=process.poll()
                    if code is not None:
                        finished.append(dict(task=task,returncode=code,finished=now()))
                        streams.pop(task).close();del active[task]
                        atomic_json(run_dir/'scheduler_status.json',dict(running={t:p.pid for t,p in active.items()},queued=queue,finished=finished,updated=now()))
                current=time.monotonic()
                if current-last_print>=10:
                    display(run_dir);last_print=current
                    atomic_json(run_dir/'scheduler_status.json',dict(running={t:p.pid for t,p in active.items()},queued=queue,finished=finished,updated=now()))
                if current-last_resource>=60:
                    append(run_dir/'resources.jsonl',resource_snapshot(active));last_resource=current
                if current-last_report>=300:
                    refresh_report();last_report=current
                if active:
                    time.sleep(.5)
        except KeyboardInterrupt:
            atomic_json(run_dir/'interruption.json',dict(time=now(),running={t:p.pid for t,p in active.items()},
                        note='Monitor interrupted; owned task processes may continue. Inspect status before resume.'))
            raise
        latency = None
        if set(tasks)==set(TASKS) and all(read_json(run_dir/t/'task_result.json',{}).get('complete') for t in TASKS):
            from .latency import measure_latency
            latency = measure_latency(run_dir,read_json(run_dir/'shared/config_resolved.json'))
        result=dict(started=started_at,finished=now(),elapsed_seconds=time.monotonic()-started,jobs=finished,
                    complete=all(read_json(run_dir/t/'task_result.json',{}).get('complete') for t in tasks),
                    serial_latency_complete=latency['complete'] if latency else None,
                    failed=[r for r in finished if r['returncode']!=0])
        atomic_json(run_dir/'run_result.json',result)
        summary=refresh_report()
        if result['complete'] and summary and summary['complete'] and set(tasks)==set(TASKS):
            from .archive import commit_results
            commit_results(run_dir,read_json(run_dir/'shared/config_resolved.json'))
        return result


def watch(run_dir,once=False):
    from open_score.research_v4.runner import read_json
    while True:
        display(run_dir)
        if once or read_json(Path(run_dir)/'run_result.json',{}).get('complete'):
            return
        time.sleep(10)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','calibrate','freeze','run-all','task','watch','summarize'])
    parser.add_argument('--run-dir',default='outputs/v5_parallel/v5_20260907_main')
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--parallel-tasks',type=int,default=6)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--multiplier',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=20260907)
    parser.add_argument('--seconds',type=float,default=900.)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--config',help='YAML overrides used only by prepare before budget freeze')
    args=parser.parse_args(argv)
    configure_threads()
    path=Path(args.run_dir).resolve()
    if args.command=='prepare':
        config=defaults(args.smoke,args.multiplier,args.seed)
        if args.config:
            import yaml
            overrides=yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
            def merge(target,values):
                for key,value in values.items():
                    if isinstance(value,dict) and isinstance(target.get(key),dict):
                        merge(target[key],value)
                    else:
                        target[key]=value
            merge(config,overrides)
        prepare(path,config,freeze=args.smoke)
    elif args.command=='freeze':
        from open_score.research_v4.runner import read_json
        prepare(path,read_json(path/'shared/config_resolved.json'),freeze=True)
    elif args.command=='task':
        if not args.task:
            parser.error('--task is required')
        execute_task(path,args.task)
    elif args.command=='run-all':
        result=run_all(path,args.parallel_tasks,[args.task] if args.task else TASKS)
        return int(not result['complete'])
    elif args.command=='watch':
        watch(path,args.once)
    elif args.command=='summarize':
        from .reporting import summarize
        summarize(path)
    elif args.command=='calibrate':
        from .calibration import calibrate
        calibrate(path,args.seconds)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
