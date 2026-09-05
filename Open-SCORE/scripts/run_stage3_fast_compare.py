"""Run the registered six-method paired comparison with resumable physical units.

Parent process deliberately imports no Torch, so four workers do not coexist
with an unnecessary coordinator copy of the models or a CUDA training cache.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import yaml

PROJECT = Path(__file__).resolve().parents[1]
_WORKER = None
METHODS = ("balanced","legacy","belief_short","belief_recourse","known_short","known_recourse")


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value)+b"\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def read_unit(path,expected):
    try:
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            value=json.load(stream)
        if value["input_hash"]!=expected or value["result_sha256"]!=digest(value["result"]):
            return None
        return value["result"]
    except (OSError,ValueError,KeyError,TypeError,EOFError):
        return None


def source_manifest():
    files=sorted((PROJECT/"src").rglob("*.py"))
    files += [Path(__file__),PROJECT/"scripts/build_stage123_report.py"]
    return {p.relative_to(PROJECT).as_posix():file_hash(p) for p in files}


def register(config):
    def seed_for(*parts):
        return int.from_bytes(hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(",",":")).encode()).digest()[:4],"little")
    result=[]
    for repeat in range(config["episodes_per_cell"]):
        for scenario in config["scenarios"]:
            for lower in config["lower_policies"]:
                for upper in config["closed_upper"]:
                    seed=seed_for("fast-paired-v2",config["seed"],scenario["label"],lower,upper,repeat)
                    for method in METHODS:
                        result.append({"id":f"fast_{scenario['label']}_{lower}_{upper}_{repeat:02d}_{method}",
                                       "scenario":scenario,"lower":lower,"upper":upper,"repeat":repeat,"seed":seed,"method":method})
    return result


def initialize(config,root):
    global _WORKER
    sys.path.insert(0,str(PROJECT/"src"))
    import torch
    from open_score.stage3.payoff import FrozenStage2Payoff
    from open_score.stage3.runtime import load_round01_stage1_model
    torch.set_num_threads(config["torch_threads_per_worker"])
    device=torch.device(config["inference_device"])
    assets=config["artifacts"]
    predictor=FrozenStage2Payoff(Path(root)/assets["stage2"],device=device,expected_sha256=assets["stage2_sha256"])
    stage1=load_round01_stage1_model(Path(root)/assets["stage1"],device,expected_sha256=assets["stage1_sha256"])
    _WORKER=(config,predictor,stage1,device)


def work(cell,path,expected):
    from open_score.stage3.unknown_upper.fast_evaluation import execute
    from open_score.stage3.unknown_upper.storage import write_unit
    config,predictor,stage1,device=_WORKER
    result=execute(cell,config,predictor,stage1,device)
    write_unit(path,expected,result)
    times=[e["planning_seconds"] for e in result["events"]]
    return {"id":cell["id"],"win":result["red_win"],"steps":result["steps"],
            "wall_seconds":result["wall_seconds"],"decision_mean":sum(times)/len(times)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=PROJECT/"configs/stage3_fast_compare.yaml")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:
        parser.error("workers must be 1..8")
    config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if tuple(config["methods"])!=METHODS or config["expected_episodes"]!=720 or config["episodes_per_cell"]!=5:
        raise ValueError("This version requires all 720 paired units, not a selected subset")
    if [(s["red"],s["blue"],s["targets"]) for s in config["scenarios"]]!=[(18,12,2),(24,16,4),(30,20,5)]:
        raise ValueError("All three physical scales are required")
    if config["lower_policies"]!=["rush","split_rush"] or config["closed_upper"]!=["balanced","concentrated_nearest","two_front","strategic_equilibrium"]:
        raise ValueError("Both known controls and all four episode-fixed upper types are required")
    root=(PROJECT/config["output_dir"]).resolve()
    if root != PROJECT/"outputs/stage123_unknown_upper_v1":
        raise ValueError("Use the single formal evidence root")
    destination=(root/config["result_subdir"]).resolve()
    destination.relative_to(root)
    destination.mkdir(parents=True,exist_ok=True)
    lock=(root/"pipeline.lock").open("a+")
    lock.seek(0);lock.write(" ");lock.flush();lock.seek(0)
    import msvcrt
    try:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        raise RuntimeError("An existing pipeline is still running; do not launch a duplicate")
    started=time.perf_counter()
    def now():return datetime.now(timezone.utc).isoformat()
    def log(message):
        line=f"[{now()}] FAST {message}"
        with (root/"pipeline.log").open("a",encoding="utf-8") as f:f.write(line+"\n")
        print(line,flush=True)
    def status(**extra):
        value={"status":"running","phase":"fast_compare_v2","pid":os.getpid(),"workers":args.workers,
               "expected":720,"updated_at":now(),**extra}
        atomic_json(root/"status.json",value);atomic_json(destination/"status.json",value)
    try:
        status(completed=0)
        assets={}
        for name in ("stage1","stage2","stage2_data"):
            path=root/config["artifacts"][name]
            actual=file_hash(path)
            if actual!=config["artifacts"][name+"_sha256"]:
                raise ValueError(f"Frozen {name} hash mismatch; no retraining or substitution")
            assets[name]={"path":str(path),"sha256":actual}
        sources=source_manifest()
        versions={p:importlib.metadata.version(p) for p in ("numpy","scipy","torch","PyYAML")}
        # Deadline behavior depends on execution load: do not silently mix worker
        # settings when resuming a time-budgeted comparison.
        context={"config":config,"assets":assets,"sources":sources,"packages":versions,"workers":args.workers}
        protocol=destination/"protocol.json"
        if protocol.exists():
            old=json.loads(protocol.read_text(encoding="utf-8"))
            if old["input_context_hash"]!=digest(context):
                raise ValueError("Inputs changed. Preserve this run and register a new result_subdir/version.")
        atomic_json(protocol,{"input_context_hash":digest(context),"context":context,
                              "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip(),
                              "created_at":now(),"meaning":"six methods on identical scenario/lower/upper/seed; exploratory fixed-protocol validation"})
        def update_root_manifest(run_status):
            path=root/"manifest.json"
            manifest=json.loads(path.read_text(encoding="utf-8-sig"))
            manifest["unknown_upper_results"]="v1_superseded_partial; v2_"+run_status
            manifest["active_stage3"]={"version":config["schema_version"],"status":run_status,
                "input_context_hash":digest(context),"protocol":str(protocol.relative_to(root)),
                "expected_episodes":720,"methods":list(METHODS),"updated_at":now(),
                "result_manifest":str((destination/"manifest.json").relative_to(root)) if run_status=="complete" else None}
            if run_status=="complete":
                manifest["historical_evidence_only"]=False
                manifest["active_stage3"]["result_manifest_sha256"]=file_hash(destination/"manifest.json")
            atomic_json(path,manifest)
        update_root_manifest("running")
        cells=register(config)
        if len(cells)!=720 or len({c["id"] for c in cells})!=720:
            raise ValueError("Invalid registry")
        atomic_json(destination/"registry.json",cells)
        units=destination/"units";units.mkdir(exist_ok=True)
        tasks=[];paths=[]
        for cell in cells:
            path=units/(cell["id"]+".json.gz")
            expected=digest({"context":digest(context),"cell":cell})
            paths.append((path,expected))
            if read_unit(path,expected) is None:
                tasks.append((cell,path,expected))
        completed=len(cells)-len(tasks);errors=[]
        log(f"{completed}/720 verified units reused; {len(tasks)} pending, workers={args.workers}; no model training")
        status(completed=completed,reused=completed,failed=0)
        # Update the single reader report before dispatch; it remains explicit
        # that formal effectiveness is unknown until all units are present.
        from build_stage123_report import build
        build(root)
        process_times=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(config,str(root))) as pool:
            futures={pool.submit(work,*task):task[0]["id"] for task in tasks}
            for future in as_completed(futures):
                name=futures[future]
                try:
                    row=future.result()
                    completed+=1;process_times.append(row["wall_seconds"])
                    log(f"{completed}/720 {name} steps={row['steps']} win={row['win']} wall={row['wall_seconds']:.1f}s")
                except Exception as error:
                    errors.append({"id":name,"error":repr(error),"traceback":traceback.format_exc()})
                    atomic_json(destination/"errors.json",errors)
                    log(f"FAILED {name}: {error}")
                recent=process_times[-48:]
                eta=(720-completed)*sum(recent)/max(1,len(recent))/args.workers if recent else None
                status(completed=completed,failed=len(errors),estimated_remaining_seconds=eta,
                       elapsed_seconds=time.perf_counter()-started,estimate_scope="recent completed units; changes with later condition difficulty")
        if errors:
            raise RuntimeError(f"{len(errors)} units failed; completed hash-valid units retained")
        if source_manifest()!=sources:
            raise RuntimeError("Code changed during execution; merge refused, units retained")
        rows=[read_unit(path,expected) for path,expected in paths]
        if any(r is None for r in rows):
            raise ValueError("Missing or corrupted result")
        sys.path.insert(0,str(PROJECT/"src"))
        from open_score.stage3.unknown_upper.fast_analysis import summarize
        summary=summarize(rows,config,destination)
        atomic_json(destination/"manifest.json",{"complete":True,"input_context_hash":digest(context),
                    "units":[{"path":str(p.relative_to(root)),"sha256":file_hash(p)} for p,_ in paths],
                    "elapsed_seconds":time.perf_counter()-started})
        build(root)
        update_root_manifest("complete")
        status(status="complete",completed=720,failed=0,elapsed_seconds=time.perf_counter()-started,
               acceptance=summary["acceptance"])
        log("All 720 paired units complete; report and CSV updated")
    except BaseException as error:
        if "update_root_manifest" in locals():
            update_root_manifest("failed")
        status(status="failed",error=repr(error),elapsed_seconds=time.perf_counter()-started)
        log(f"STOPPED: {error}")
        raise
    finally:
        lock.close()


if __name__=="__main__":
    main()
