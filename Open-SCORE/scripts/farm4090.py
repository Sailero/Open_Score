#!/usr/bin/env python3
"""Pack remaining main0923 jobs (skip trial) and run them on 4090 boxes.

The official pipeline is a single-node 2-GPU scheduler. This script does not
replace it and does not touch a live main0923 run. Each 4090 machine gets its
own output copy and a static shard of the remaining train list, starting from
step 0 (no resume.pt). After trains finish, copy new finals onto one machine
and run eval there.

  python scripts/farm4090.py pack --source outputs/main0923 --dest outputs/pack4090
  python scripts/farm4090.py jobs --output outputs/pack4090
  python scripts/farm4090.py train --output $OUT --devices 0,1,2,3 --per-gpu 1 --env had --shard 0/2
  python scripts/farm4090.py eval --output $OUT --devices 0,1 --env had
  python scripts/farm4090.py merge --from $MACHINE_OUT --into $MASTER_OUT
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from multiprocessing import get_context
import os
from pathlib import Path
from queue import Empty
import shutil
import signal
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT_FILES = (
    "experiment.json", "implementation_acceptance.json", "pipeline.json",
    "episodes.csv", "learning.csv", "progress.csv", "matched_hidden.json",
)
ROOT_DIRS = ("decision", "probe", "smacv2/scenes")
CHECKPOINT_FILES = ("final.pt", "best.pt", "config.json", "matched_hidden.json")
TRIAL_HAD = (
    ("regir_kv0_intent_sg", (0, 1, 2, 3, 4)),
    ("regir_kv0_sg", (0, 1, 2, 3, 4)),
    ("regir_r1_sg", (0, 1, 2, 3, 4)),
    ("regir_r0_sg", (0, 1, 2)),
)
PRIORITY = ("common:P0", "branch:A:P0", "common:P1", "branch:A:P1", "common:P2", "branch:A:P2")


def _stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _parse_devices(text):
    ids = [int(part) for part in str(text).split(",") if part.strip() != ""]
    if not ids or len(set(ids)) != len(ids) or any(index < 0 for index in ids):
        raise SystemExit(f"invalid --devices {text!r}")
    return tuple(ids)


def _parse_shard(text):
    if text in (None, "", "all"):
        return 0, 1
    left, sep, right = str(text).partition("/")
    if sep != "/" or not left.isdigit() or not right.isdigit():
        raise SystemExit("--shard must look like 0/3")
    index, count = int(left), int(right)
    if count < 1 or not 0 <= index < count:
        raise SystemExit(f"--shard {text} is outside 0..{count - 1}")
    return index, count


def remaining_trains(branch="A"):
    from open_score.eval import experiment as X
    jobs = []
    for env in ("had", "smacv2"):
        for (method, seed), entry in X.train_matrix(env).items():
            segments = entry["segments"]
            if "trial" in segments or "imported" in segments:
                continue
            if env == "had" and method in X.IMPORTED_BASELINES:
                continue
            keep = {segment: priority for segment, priority in segments.items()
                    if segment == "common" or segment == f"branch:{branch}"}
            if not keep:
                continue
            keys = []
            for segment, priority in keep.items():
                keys.append("common:" + priority if segment == "common" else f"{segment}:{priority}")
            rank = min(PRIORITY.index(key) if key in PRIORITY else len(PRIORITY) for key in keys)
            jobs.append(dict(id=f"train.{env}.{method}.s{seed}", env=env, method=method,
                             seed=int(seed), segments=keep, rank=rank, order=int(entry["order"]),
                             t_max=X.budget(env)))
    jobs.sort(key=lambda row: (row["rank"], row["order"], row["id"]))
    return jobs


def _run_dir(output, job):
    from open_score.eval.experiment import run_directory
    return run_directory(output, job["method"], job["seed"], job["env"])


def _final_ready(output, job):
    from open_score.eval.experiment import checkpoint_info
    path = _run_dir(output, job) / "final.pt"
    if not path.is_file():
        return False
    try:
        checkpoint_info(path, method=job["method"], seed=job["seed"], env=job["env"])
        return True
    except (ValueError, KeyError, OSError):
        return False


def _copy_file(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _copy_tree(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


def _kept_runs(source):
    from open_score.eval import experiment as X
    from open_score.eval.experiment import run_directory
    rows = []
    for method, seeds in TRIAL_HAD:
        for seed in seeds:
            rows.append(run_directory(source, method, seed, "had"))
    for method in (*X.IMPORTED_BASELINES, *X.IMPORTED_NO_SG):
        for seed in X.SEEDS:
            path = run_directory(source, method, seed, "had")
            if (path / "final.pt").is_file():
                rows.append(path)
    return rows


def cmd_pack(options):
    source, dest = options.source.resolve(), options.dest.resolve()
    if dest.exists() and any(dest.iterdir()) and not options.force:
        raise SystemExit(f"{dest} is not empty; pass --force to overwrite files in place")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"{_stamp()} pack   {source} -> {dest}", flush=True)
    for name in ROOT_FILES:
        src = source / name
        if src.exists():
            _copy_file(src, dest / name)
            print(f"  file  {name}", flush=True)
    for name in ROOT_DIRS:
        src = source / name
        if src.is_dir():
            _copy_tree(src, dest / name)
            print(f"  dir   {name}", flush=True)
    for directory in _kept_runs(source):
        relative = directory.relative_to(source)
        for name in CHECKPOINT_FILES:
            src = directory / name
            if src.exists():
                _copy_file(src, dest / relative / name)
        print(f"  ckpt  {relative}", flush=True)
    jobs = remaining_trains("A")
    manifest = dict(
        created_at=_stamp(), source=str(source), branch="A", skip="trial",
        resume=False, trains=jobs,
        note="Do not copy remaining train directories or any resume.pt. "
             "Do not start official run_all.sh on this pack; use farm4090.py.")
    (dest / "remaining_jobs.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (dest / "4090说明.md").write_text(_readme(), encoding="utf-8")
    print(f"{_stamp()} pack   {len(jobs)} remaining trains written to remaining_jobs.json", flush=True)
    print(f"{_stamp()} pack   done. rsync this directory plus the Open-SCORE repo.", flush=True)
    return 0


def cmd_jobs(options):
    output = options.output.resolve()
    jobs = remaining_trains("A")
    if options.env != "all":
        jobs = [job for job in jobs if job["env"] == options.env]
    index, count = _parse_shard(options.shard)
    jobs = jobs[index::count]
    for job in jobs:
        mark = "done" if _final_ready(output, job) else "todo"
        print(f"{mark:4s}  {job['id']:<42}  rank={job['rank']}  t_max={job['t_max']}")
    print(f"# {sum(1 for job in jobs if not _final_ready(output, job))}/{len(jobs)} remaining "
          f"env={options.env} shard={index}/{count}")
    return 0


def _job_entry(job, output, stop_event, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from open_score.algos import train
    from open_score.eval.experiment import PROFILE, run_directory
    directory = run_directory(output, job["method"], job["seed"], job["env"])
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(profile=PROFILE, output=str(output), env=job["env"], run="train",
                   seed=int(job["seed"]), t_max=int(job["t_max"]),
                   batch_size_run=8 if job["env"] == "had" else 4,
                   use_cuda=True, resume=False, skip_final_eval=True,
                   reward_mode="damage", friendly_penalty=1.0, _stop_event=stop_event)
    if job["env"] == "smacv2":
        sc2 = os.environ.get("SC2PATH")
        if not sc2:
            results.put(dict(id=job["id"], status="failed", error="SC2PATH is not set"))
            return
        payload["env_args"] = {"sc2path": sc2}
    with (directory / "console.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                result = train(job["method"], payload)
                status = "completed" if Path(result).name == "final.pt" else "stopped"
                results.put(dict(id=job["id"], status=status, checkpoint=str(result)))
            except BaseException:
                print(traceback.format_exc(), flush=True)
                results.put(dict(id=job["id"], status="failed", error=traceback.format_exc()[-4000:]))


def cmd_train(options):
    from open_score.eval.experiment import initialize
    output = options.output.resolve()
    initialize(output)
    devices = _parse_devices(options.devices)
    jobs = remaining_trains("A")
    if options.env != "all":
        jobs = [job for job in jobs if job["env"] == options.env]
    index, count = _parse_shard(options.shard)
    jobs = [job for job in jobs[index::count] if not _final_ready(output, job)]
    if options.env in ("smacv2", "all") and not os.environ.get("SC2PATH"):
        print("warning: SC2PATH is empty; SMAC jobs will fail", flush=True)
    slots = min(options.max_jobs, len(devices) * options.per_gpu)
    print(f"{_stamp()} train  devices={','.join(map(str, devices))}  per_gpu={options.per_gpu}  "
          f"max_jobs={options.max_jobs}  slots={slots}  shard={index}/{count}  env={options.env}  "
          f"queue={len(jobs)}  resume=False",
          flush=True)
    if not jobs:
        print(f"{_stamp()} train  nothing to do", flush=True)
        return 0
    context = get_context("spawn")
    stop_event, results = context.Event(), context.Queue()
    live, waiting = [], list(jobs)
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    def load():
        counts = {gpu: 0 for gpu in devices}
        for item in live:
            counts[item["gpu"]] += 1
        return counts

    def pick():
        if len(live) >= options.max_jobs:
            return None
        free = [(count, gpu) for gpu, count in load().items() if count < options.per_gpu]
        return min(free)[1] if free else None

    def launch():
        while waiting and not stop_event.is_set():
            gpu = pick()
            if gpu is None:
                break
            job = dict(waiting.pop(0), gpu=gpu)
            child = context.Process(target=_job_entry, args=(job, str(output), stop_event, results))
            child.start()
            live.append(dict(child=child, job=job, gpu=gpu, started=time.monotonic()))
            print(f"{_stamp()} start  {job['id']}  gpu={gpu}  live={len(live)}/{slots}  "
                  f"queue={len(waiting)}", flush=True)

    failed = []
    launch()
    while live:
        try:
            payload = results.get(timeout=1)
        except Empty:
            payload = None
        if payload:
            identity = payload["id"]
            item = next(row for row in live if row["job"]["id"] == identity)
            item["child"].join(timeout=30)
            live.remove(item)
            print(f"{_stamp()} {payload['status']:9s} {identity}  live={len(live)}/{slots}  "
                  f"queue={len(waiting)}", flush=True)
            if payload["status"] == "failed":
                failed.append(identity)
                err = payload.get("error") or ""
                print(err[-500:], flush=True)
        for item in list(live):
            if item["child"].is_alive():
                continue
            if any(row["job"]["id"] == item["job"]["id"] for row in live) and item in live:
                live.remove(item)
        if not stop_event.is_set():
            launch()
    print(f"{_stamp()} train  finished  failed={len(failed)}", flush=True)
    return 1 if failed else 0


def _load_eval_module():
    import importlib.util
    path = PROJECT / "scripts/eval.py"
    spec = importlib.util.spec_from_file_location("farm4090_eval_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_eval(options):
    from open_score.eval.experiment import HAD_EVAL_KINDS, initialize, atomic_json
    run_eval = _load_eval_module().run_eval
    output = options.output.resolve()
    initialize(output)
    devices = _parse_devices(options.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    allowed = set(HAD_EVAL_KINDS) if options.env == "had" else {"final", "depth"}
    only = ({k.strip() for k in options.only.split(",")} if options.only else allowed)
    if not only <= allowed:
        raise SystemExit(f"unsupported eval kinds for {options.env}: {sorted(only - allowed)}")
    shards = 96 if options.env == "had" else 32
    if options.final_shards:
        shards = options.final_shards
    manifest = initialize(output)
    manifest.setdefault("resources", {}).setdefault("evaluation", {})[options.env] = dict(
        device="auto", max_concurrent=options.max_concurrent, final_shards=shards,
        physical_gpus=list(devices), gpu_workers_per_device=options.gpu_workers_per_device,
        gpu_reserve_gib=8, updated_at=time.time())
    atomic_json(output / "experiment.json", manifest)
    print(f"{_stamp()} eval   env={options.env} devices={devices} only={sorted(only)}", flush=True)
    ok = run_eval(output, max_concurrent=options.max_concurrent, only=only, device="auto",
                  env=options.env, final_shards=shards, devices=devices,
                  gpu_workers_per_device=options.gpu_workers_per_device)
    return 0 if ok else 1


def cmd_merge(options):
    src, dest = options.source.resolve(), options.dest.resolve()
    jobs = remaining_trains("A")
    copied = 0
    for job in jobs:
        origin, target = _run_dir(src, job), _run_dir(dest, job)
        if not (origin / "final.pt").is_file():
            continue
        if _final_ready(dest, job) and not options.force:
            continue
        target.mkdir(parents=True, exist_ok=True)
        for name in CHECKPOINT_FILES + ("console.log",):
            if (origin / name).exists():
                _copy_file(origin / name, target / name)
        copied += 1
        print(f"  merge {job['id']}", flush=True)
    print(f"{_stamp()} merge  copied {copied} new finals from {src}", flush=True)
    print("Append new evaluation rows yourself, or re-run eval on the merged tree.", flush=True)
    return 0


def _readme():
    jobs = remaining_trains("A")
    had = [job for job in jobs if job["env"] == "had"]
    smac = [job for job in jobs if job["env"] == "smacv2"]
    lines = [
        "# 4090 剩余实验（跳过试训，从 0 开训）",
        "",
        "官方 `run_all.sh` 是单机、物理卡 0/1、全局 6 槽。这个包不走那条调度。",
        "试训 18 条 HAD 和导入基线已经带上；剩余训练目录是空的，会从第 0 步开始。",
        "",
        f"- 剩余训练：{len(jobs)}（HAD {len(had)}，SMAC {len(smac)}）",
        "- 不要上传任何 `resume.pt`，也不要上传正在跑的 `smacv2/refil`",
        "",
        "## 每台机器上传什么",
        "",
        "1. 整个 `Open-SCORE` 仓库（含这个 `scripts/farm4090.py`）",
        "2. 这个 `pack4090` 目录（约 1.3G：试训 final + 导入 + episodes/learning/progress + scenes/probe/decision）",
        "3. HAD 环境 `saileron`；SMAC 另需 `saileron-smac` 和 `StarCraftII`（设 `SC2PATH`）",
        "4. **不要**上传 `outputs/main0921` 全量，导入已经在包里",
        "5. **不要**上传当前 6000 上半成品 SMAC",
        "",
        "## 多机怎么拆",
        "",
        "不要共享同一个 output 目录。每台机器一份 pack 副本，用 `--shard i/n` 或 `--env` 静态拆任务。",
        "",
        "```bash",
        "# 机器 A：2 张 4090，只跑 HAD，每卡先 1 个",
        "export REGIR_REPO=$PWD/../..   # 仓库根，含 Open-SCORE",
        "cd Open-SCORE",
        "python scripts/farm4090.py train --output $OUT --devices 0,1 --per-gpu 1 --env had --shard 0/2",
        "",
        "# 机器 B：同样 HAD 的另一半",
        "python scripts/farm4090.py train --output $OUT --devices 0,1 --per-gpu 1 --env had --shard 1/2",
        "",
        "# 机器 C：有 SC2，跑全部 SMAC（每卡先 1 个；spectra/REFIL 24GB 峰值可能装不下 2 个）",
        "export SC2PATH=/path/to/StarCraftII",
        "PYTHONNOUSERSITE=1 $SMAC_PY scripts/farm4090.py train --output $OUT --devices 0,1,2,3 --per-gpu 1 --env smacv2",
        "```",
        "",
        "训完后把各机新的 `*/train/seed_*/final.pt` rsync 回一台，再：",
        "",
        "```bash",
        "python scripts/farm4090.py merge --from $MACHINE_OUT --into $MASTER_OUT",
        "python scripts/farm4090.py eval --output $MASTER_OUT --devices 0,1 --env had",
        "PYTHONNOUSERSITE=1 $SMAC_PY scripts/farm4090.py eval --output $MASTER_OUT --devices 0,1 --env smacv2 --only final,depth",
        "```",
        "",
        "`--per-gpu` 你自己试。HAD 建议先 1；SMAC REFIL 实测 reserved 到过 24.5 GiB，4090 上 2 个通常进不去。",
        "",
    ]
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--source", type=Path, default=PROJECT / "outputs/main0923")
    pack.add_argument("--dest", type=Path, default=PROJECT / "outputs/pack4090")
    pack.add_argument("--force", action="store_true")
    jobs = sub.add_parser("jobs")
    jobs.add_argument("--output", type=Path, default=PROJECT / "outputs/pack4090")
    jobs.add_argument("--env", choices=("had", "smacv2", "all"), default="all")
    jobs.add_argument("--shard", default="all")
    train = sub.add_parser("train")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--devices", required=True)
    train.add_argument("--per-gpu", type=int, default=1)
    train.add_argument("--max-jobs", type=int, default=2,
                       help="Hard cap on concurrent trainers on this machine (default 2).")
    train.add_argument("--env", choices=("had", "smacv2", "all"), default="all")
    train.add_argument("--shard", default="all")
    ev = sub.add_parser("eval")
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--devices", default="0,1")
    ev.add_argument("--env", choices=("had", "smacv2"), required=True)
    ev.add_argument("--only", default=None)
    ev.add_argument("--max-concurrent", type=int, default=8)
    ev.add_argument("--gpu-workers-per-device", type=int, default=2)
    ev.add_argument("--final-shards", type=int, default=None)
    merge = sub.add_parser("merge")
    merge.add_argument("--from", dest="source", type=Path, required=True)
    merge.add_argument("--into", dest="dest", type=Path, required=True)
    merge.add_argument("--force", action="store_true")
    return result


def main():
    options = parser().parse_args()
    commands = dict(pack=cmd_pack, jobs=cmd_jobs, train=cmd_train, eval=cmd_eval, merge=cmd_merge)
    raise SystemExit(commands[options.command](options))


if __name__ == "__main__":
    main()
