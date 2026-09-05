"""The formal BA-DIB coordinator. No reduced grid can be labelled Formal."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import importlib.metadata
import os
import platform
from pathlib import Path
import subprocess
import sys
import traceback

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))

import torch
import yaml
from open_score.stage3.payoff import FrozenStage2Payoff
from open_score.stage3.runtime import load_round01_stage1_model
from open_score.stage3.unknown_upper.analysis import summarize
from open_score.stage3.unknown_upper.diagnostics import run_diagnostics
from open_score.stage3.unknown_upper.evaluation import ablation_registry,execute_episode,formal_registry,training_registry
from open_score.stage3.unknown_upper.planner import ABLATION_METHODS,Commander,METHODS
from open_score.stage3.unknown_upper.storage import atomic_json,digest,file_hash,read_unit,write_unit
from open_score.stage3.unknown_upper.training import load_qom,train_qom

_WORKER=None


def now():
    return datetime.now(timezone.utc).isoformat()


def cached_json(path):
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError,ValueError):
        return {}


def validate_protocol(config):
    if tuple(config["methods"])!=METHODS or config["episodes_per_cell"]!=10 or config["planning"]!={
        "candidates":8,"rerank_repeats":4,"simulations":32,"depth":2,
        "leaf_value":"terminal_rollout_public_threat_policy","observation_abstraction":"public_position250_velocity75_health_quarters"}:
        raise ValueError("Formal protocol cannot silently reduce the registered method or planning grid")
    expected={"lower_policies":["rush","split_rush"],
              "closed_upper":["balanced","concentrated_nearest","two_front","strategic_equilibrium"],
              "open_upper":[]}
    if any(config[k]!=v for k,v in expected.items()):
        raise ValueError("Formal policy families must match the registered grid")
    if [(x["red"],x["blue"],x["targets"]) for x in config["scenarios"]]!=[(18,12,2),(24,16,4),(30,20,5)]:
        raise ValueError("Formal scenarios must retain all 30/40/50-agent scales")
    physical={"max_steps":50,"command_interval":5,"replan_on_casualty":True,"blue_seed_clock":"physical_step",
              "reserve_mode":"patrol"}
    training={"trajectories_per_cell":50,"probes":["balanced","threat"],"split":[.70,.15,.15],
              "codebook":8,"latent":32}
    if any(config["physical"][k]!=v for k,v in physical.items()) or any(config["training"][k]!=v for k,v in training.items()):
        raise ValueError("Configuration must agree with implemented clock, split and belief semantics")
    if config["paper_ablations"]!={"methods":list(ABLATION_METHODS),"expected_episodes":480}:
        raise ValueError("The paper requires both matched 240-episode controls")
    if config["acceptance"]["expected_episodes"]!=1920:
        raise ValueError("Fixed-type formal core requires exactly 1,920 registered episodes")


def source_manifest():
    files=sorted((PROJECT/"src").rglob("*.py"))
    files+=[Path(__file__),PROJECT/"scripts/build_stage123_report.py"]
    return {p.relative_to(PROJECT).as_posix():file_hash(p) for p in files}


def runtime_manifest():
    versions = {package:importlib.metadata.version(package) for package in ["torch","numpy","scipy","scikit-learn","PyYAML"]}
    return {"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
            "processor":platform.processor(),"logical_cpus":os.cpu_count(),"packages":versions,
            "cuda_available":torch.cuda.is_available(),
            "cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip()}


def initialize(config,root,qom_path=None):
    global _WORKER
    torch.set_num_threads(config["torch_threads_per_worker"])
    device=torch.device(config["inference_device"])
    artifacts=config["artifacts"]
    predictor=FrozenStage2Payoff(Path(root)/artifacts["stage2"],device=device,expected_sha256=artifacts["stage2_sha256"])
    stage1=load_round01_stage1_model(Path(root)/artifacts["stage1"],device,expected_sha256=artifacts["stage1_sha256"])
    factory=load_qom(qom_path,device)[1] if qom_path else None
    commander=Commander(predictor,stage1,device,config["planning"],factory)
    _WORKER=(config,predictor,stage1,device,commander)


def work(cell,path,input_hash):
    config,predictor,stage1,device,commander=_WORKER
    result=execute_episode(cell,config,predictor,stage1,device,commander if "method" in cell else None)
    write_unit(path,input_hash,result)
    return {"id":cell["id"],"steps":result["steps"],"win":result["red_win"]}


def verify_artifacts(config,root):
    result={}
    for name in ["stage1","stage2","stage2_data"]:
        path=root/config["artifacts"][name]
        actual=file_hash(path)
        if actual!=config["artifacts"][name+"_sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch: {path}; Stage2 will NOT be retrained")
        result[name]={"path":str(path),"sha256":actual}
    manifest=json.loads((root/"stage2/data/dataset_manifest.json").read_text(encoding="utf8"))
    if manifest["episodes"]!=19200 or manifest["dataset_sha256"]!=result["stage2_data"]["sha256"]:
        raise ValueError("Stage2 manifest disagrees with the frozen formal dataset")
    return result


def run_units(cells,phase,input_context,config,root,workers,log,status,qom_path=None):
    directory=root/"stage3/unknown_upper"/("data/trajectories" if phase=="collect_qom" else "units")
    tasks=[];paths=[];entries=[]
    for cell in cells:
        path=directory/(cell["id"]+".json.gz")
        expected=digest({"context":input_context,"cell":cell})
        existing=read_unit(path,expected)
        paths.append(path)
        entries.append({"id":cell["id"],"input_hash":expected,"path":path.relative_to(root).as_posix()})
        if existing is None:
            tasks.append((cell,path,expected))
    completed=len(cells)-len(tasks)
    status(phase,completed,len(cells),reused=completed)
    log(f"{phase}: {completed}/{len(cells)} valid completed units reused; {len(tasks)} pending")
    errors=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(config,str(root),str(qom_path) if qom_path else None)) as pool:
        futures={pool.submit(work,*task):task[0]["id"] for task in tasks}
        for future in as_completed(futures):
            unit_id=futures[future]
            try:
                result=future.result()
                completed+=1
                log(f"{phase} {completed}/{len(cells)} {unit_id} steps={result['steps']} win={result['win']}")
            except Exception as error:
                errors.append({"id":unit_id,"error":repr(error),"traceback":traceback.format_exc()})
                log(f"FAILED {unit_id}: {error}")
                atomic_json(root/"stage3/unknown_upper/unit_errors.json",errors)
            status(phase,completed,len(cells),failed=len(errors))
    if errors:
        raise RuntimeError(f"{len(errors)} registered units failed; valid finished units are resumable")
    for entry,path in zip(entries,paths):
        if read_unit(path,entry["input_hash"]) is None:
            raise ValueError(f"Post-merge hash validation failed: {path}")
        entry["file_sha256"]=file_hash(path)
    atomic_json(root/"stage3/unknown_upper"/(phase+"_manifest.json"),{"units":entries,"context":input_context,"complete":True})
    return paths,entries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=PROJECT/"configs/stage123_unknown_upper.yaml")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:
        parser.error("workers must be positive")
    config=yaml.safe_load(args.config.read_text(encoding="utf8"))
    validate_protocol(config)
    root=(PROJECT/config["output_dir"]).resolve()
    if root!=PROJECT/"outputs/stage123_unknown_upper_v1":
        raise ValueError("The formal pipeline has one registered outputs root")
    root.mkdir(parents=True,exist_ok=True)
    # OS file locking prevents two coordinators from racing the same units.
    lock=(root/"pipeline.lock").open("a+")
    lock.seek(0);lock.write(" ");lock.flush();lock.seek(0)
    import msvcrt
    try:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        raise RuntimeError("A coordinator already holds the formal pipeline lock")
    started=now()
    def log(message):
        line=f"[{now()}] {message}"
        with (root/"pipeline.log").open("a",encoding="utf8") as stream:
            stream.write(line+"\n")
        print(line,flush=True)
    def status(phase,completed=0,expected=2400,**extra):
        atomic_json(root/"status.json",{"status":"running","phase":phase,"completed":completed,"expected":expected,
                                        "pid":os.getpid(),"workers":args.workers,"started_at":started,"updated_at":now(),**extra})
    try:
        status("validate")
        log("Formal BA-DIB pipeline starting; no Stage2 collection/training is allowed")
        artifacts=verify_artifacts(config,root)
        tests=subprocess.run([sys.executable,"-m","pytest","-q","tests/test_unknown_upper.py"],cwd=PROJECT,text=True,encoding="utf8",stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log(tests.stdout)
        if tests.returncode:
            raise RuntimeError("Repository tests failed before formal evaluation")
        sources=source_manifest()
        runtime=runtime_manifest()
        atomic_json(root/"stage3/unknown_upper/protocol.json",{"config":config,"sources":sources,"artifacts":artifacts,"runtime":runtime})
        train_cells=training_registry(config)
        formal_cells=formal_registry(config)
        ablation_cells=ablation_registry(config)
        atomic_json(root/"stage3/unknown_upper/registry.json",{"training":train_cells,"formal":formal_cells,"paper_ablations":ablation_cells})
        # Include the planner module because it owns the training probe's tail
        # action. Reporting-only edits do not invalidate physical trajectories.
        collection_sources={k:v for k,v in sources.items() if k.startswith(("src/HAD_Env/","src/open_score/envs/","src/open_score/stage1/","src/open_score/stage2/")) or k.endswith(("unknown_upper/domain.py","unknown_upper/world.py","unknown_upper/policies.py","unknown_upper/planner.py","unknown_upper/evaluation.py","stage3/runtime.py","stage3/payoff.py"))}
        collection_context={"sources":collection_sources,"artifacts":artifacts,"physical":config["physical"],
                            "packages":runtime["packages"],"seed":config["seed"]}
        training_paths,training_entries=run_units(train_cells,"collect_qom",collection_context,config,root,args.workers,log,status)
        model_dir=root/"stage3/unknown_upper/model"
        model_path=model_dir/"qom.pt"
        model_manifest=model_dir/"manifest.json"
        model_input=digest({"data":[(e["id"],e["file_sha256"]) for e in training_entries],"config":config["training"],"seed":config["seed"],
                            "packages":runtime["packages"],
                            "sources":{k:v for k,v in sources.items() if k.endswith(tuple("unknown_upper/"+name+".py" for name in ["training","qom","belief","domain","policies","world"]))}})
        old=cached_json(model_manifest)
        metrics_path=model_dir/"training_metrics.json"
        reuse=(model_path.exists() and metrics_path.exists() and old.get("input_hash")==model_input
               and old.get("checkpoint_sha256")==file_hash(model_path)
               and old.get("metrics_sha256")==file_hash(metrics_path))
        initialize(config,root)
        _,predictor,stage1,device,_=_WORKER
        if not reuse:
            status("train_qom")
            metrics=train_qom(training_paths,config,model_dir,predictor,log)
            atomic_json(model_manifest,{"input_hash":model_input,"checkpoint_sha256":file_hash(model_path),"metrics_sha256":file_hash(model_dir/"training_metrics.json")})
        else:
            log("QOM checkpoint and its data/config/source hashes match; reuse")
        initialize(config,root,model_path)
        _,predictor,stage1,device,commander=_WORKER
        status("diagnostics")
        diagnostic_context=digest({"sources":sources,"model":file_hash(model_path),"config":config,"packages":runtime["packages"]})
        diagnostic_path=root/"stage3/unknown_upper/diagnostics.json"
        diagnostics=cached_json(diagnostic_path)
        diagnostic_manifest=diagnostic_path.with_name("diagnostics_manifest.json")
        valid_diagnostics=(diagnostic_path.exists() and cached_json(diagnostic_manifest).get("sha256")==file_hash(diagnostic_path))
        if not valid_diagnostics or diagnostics.get("input_hash")!=diagnostic_context or not diagnostics.get("hard_checks_passed"):
            diagnostics=run_diagnostics(config,root/"stage3/unknown_upper",predictor,stage1,device,commander,training_paths,log)
            diagnostics["input_hash"]=diagnostic_context
            atomic_json(diagnostic_path,diagnostics)
            atomic_json(diagnostic_manifest,{"sha256":file_hash(diagnostic_path)})
        context={"config":config,"sources":sources,"artifacts":artifacts,"packages":runtime["packages"],
                 "qom_sha256":file_hash(model_path),"diagnostics_sha256":file_hash(diagnostic_path)}
        paths,entries=run_units(formal_cells,"formal",context,config,root,args.workers,log,status,model_path)
        ablation_paths,ablation_entries=run_units(ablation_cells,"paper_ablations",context,config,root,args.workers,log,status,model_path)
        if source_manifest()!=sources:
            raise RuntimeError("Implementation changed during execution; merge refused, units retained")
        summary=summarize(paths,config,root/"stage3/unknown_upper",diagnostics,ablation_paths)
        from build_stage123_report import build
        build(root)
        manifest=json.loads((root/"manifest.json").read_text(encoding="utf8"))
        manifest.update({"historical_evidence_only":False,"unknown_upper_results":"complete","formal_protocol_sha256":file_hash(root/"stage3/unknown_upper/protocol.json"),
                         "formal_registry_sha256":file_hash(root/"stage3/unknown_upper/registry.json"),"qom_sha256":file_hash(model_path),
                         "formal_summary_sha256":file_hash(root/"stage3/unknown_upper/summary.json"),"formal_manifest_sha256":file_hash(root/"stage3/unknown_upper/formal_manifest.json")})
        atomic_json(root/"manifest.json",manifest)
        atomic_json(root/"status.json",{"status":"completed","phase":"complete","completed":len(entries),"expected":1920,
                                        "paper_ablation_completed":len(ablation_entries),"paper_ablation_expected":480,"all_acceptance_passed":summary["all_passed"],
                                        "started_at":started,"completed_at":now(),"pid":os.getpid(),"report":str(root/"experiment_report.md")})
        log(f"All 1,920 core and 480 matched control units completed. Acceptance passed={summary['all_passed']}")
    except Exception as error:
        previous=json.loads((root/"status.json").read_text(encoding="utf8"))
        previous.update({"status":"failed","failed_phase":previous.get("phase"),"phase":"failed","error":repr(error),"updated_at":now()})
        atomic_json(root/"status.json",previous)
        log(traceback.format_exc())
        raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=="__main__":
    main()
