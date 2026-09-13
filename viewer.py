"""Local HAD replay and comparison UI for rules and retained checkpoints."""
from __future__ import annotations

import argparse
import copy
import errno
import importlib
import importlib.util
import json
import math
import os
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Open-SCORE"))
HTML = ROOT / "viewer.html"
OUTPUTS = ROOT / "Open-SCORE" / "outputs"
JOBS = {}
LOCK = threading.Lock()
CATALOG_LOCK = threading.RLock()
# ALMA's evaluation proposals use the CPU torch RNG. Keep jobs sequential,
# and seed proposals per episode independently of model construction.
RUN_LOCK = threading.Lock()
POLICY_CONTEXT = threading.local()
CHECKPOINTS = {}
CUSTOM_PATHS = set()
CATALOG_READY = False
CHECKPOINT_POLICY = "_viewer_checkpoint"
METHOD_LABELS = {
    "b0_qmix": "QMIX-Base", "b2_qmix_atten": "QMIX-Atten",
    "refil": "REFIL", "dcg": "DCG", "gnn_qmix": "GNN-QMIX",
    "spectra": "SPECTra", "alma": "ALMA",
}
DEFAULTS = {
    "targets": 2, "red": 8, "blue": 8, "seed0": 0,
    "display": 5, "stats": 100,
    "task_mode": "damage",
    "spatial_dim": 2,
    "target_initialization": "random",
}
MAX_SAFE_INTEGER = 2 ** 53 - 1


def _path_id(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _describe_checkpoint(path, cfg):
    from open_score.algos import METHODS
    from open_score.envs.features import MAX_AGENTS, MAX_BLUE, MAX_TARGETS, ENTITY_DIM, N_ACTIONS
    legacy_name = path.parent.parent.name if path.parent.name == "models" else path.stem
    method = cfg.get("method", legacy_name)
    try:
        version = path.relative_to(OUTPUTS).parts[0]
    except ValueError:
        version = "自定义"
    run, seed = cfg.get("run", path.parent.name), cfg.get("seed")
    limits = dict(red=MAX_AGENTS, blue=MAX_BLUE, targets=MAX_TARGETS,
                  max_steps=int(cfg.get("episode_limit", 100)))
    budget = cfg.get("pool_slots")
    if budget is not None:
        if not isinstance(budget, (list, tuple)) or len(budget) != 3 or any(int(n) < 1 for n in budget):
            raise ValueError("checkpoint 的 pool_slots 必须包含三个正整数。")
        limits.update({key: min(limits[key], int(n)) for key, n in zip(("red", "blue", "targets"), budget)})
    reason = ""
    if path.name == "resume.pt":
        reason = "训练恢复文件（包含 replay）；请选择 best、final 或 latest 推理权重。"
    elif method not in METHODS:
        reason = "历史或未知模型格式；需要对应算法的推理实现，不能直接作为当前动作策略加载。"
    elif cfg.get("env") != "had" or cfg.get("feature_layout") != "had":
        reason = "该权重不是 HAD 实体观测模型（原生 E0 环境不能用于 HAD 回放）。"
    elif int(cfg.get("n_actions", 0)) != N_ACTIONS or int(cfg.get("entity_shape", 0)) != ENTITY_DIM:
        reason = "观测或动作维度与当前 HAD checkpoint 接口不一致。"
    elif cfg.get("encoder") == "flatten" and budget is None and int(cfg.get("n_entities", 0)) != MAX_AGENTS + MAX_BLUE + MAX_TARGETS:
        reason = "旧有序展平模型没有 pool_slots，固定输入宽度与当前实体接口不兼容。"
    elif cfg.get("multi_task") and int(cfg.get("n_tasks", 0)) + int(cfg.get("n_extra_tasks", 0)) < MAX_TARGETS:
        reason = "分层模型的子任务嵌入不足以重建当前 6 目标槽接口。"
    label = f"{METHOD_LABELS.get(method, method)} · {version} / {run} · seed {seed if seed is not None else '未知'} · {path.stem}"
    return dict(id=_path_id(path), path=_path_id(path), label=label, method=method,
                method_label=METHOD_LABELS.get(method, method), version=version,
                run=run, seed=seed, kind=path.stem, available=not reason, reason=reason,
                limits=limits, task_mode="damage", spatial_dim=2,
                architecture="hierarchical" if cfg.get("multi_task") else "end_to_end")


def _scan_checkpoints():
    """Read sidecar configs, not replay buffers or all network tensors."""
    global CATALOG_READY
    with CATALOG_LOCK:
        paths = set(OUTPUTS.rglob("*.pt")) | set((ROOT / "Open-SCORE/assets/frozen").glob("*.pt")) | CUSTOM_PATHS
        found = {}
        configs = {}
        for path in sorted(paths):
            path = path.resolve()
            identity = _path_id(path)
            try:
                if path in CUSTOM_PATHS and identity in CHECKPOINTS:
                    row = dict(CHECKPOINTS[identity])
                    stat = path.stat()
                    if row.get("file_size") != stat.st_size or row.get("modified_ns") != str(stat.st_mtime_ns):
                        row = _register_checkpoint({"path": str(path)})
                else:
                    config_path = path.parent / "config.json"
                    if config_path not in configs:
                        configs[config_path] = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
                    row = _describe_checkpoint(path, configs[config_path])
                    stat = path.stat()
                    row.update(file_size=stat.st_size, modified_ns=str(stat.st_mtime_ns))
            except Exception as error:
                row = dict(id=identity, path=identity, label=identity, method="unknown",
                           method_label="未知模型", version="未识别", run="", seed=None, kind=path.stem,
                           available=False, reason=f"无法读取权重配置：{error}", limits={}, task_mode="damage", spatial_dim=2)
            found[identity] = row
        CHECKPOINTS.clear()
        CHECKPOINTS.update(found)
        CATALOG_READY = True


def _checkpoint_factory(seed):
    import torch
    policy = getattr(POLICY_CONTEXT, "policy", None)
    if policy is None:
        raise ValueError("viewer checkpoint 必须在评估任务内加载。")
    torch.manual_seed(int(seed) % (2 ** 63))
    return policy


def _checkpoint_row(identity):
    if not isinstance(identity, str):
        raise ValueError("请选择一个已发现或手动添加的 checkpoint。")
    with CATALOG_LOCK:
        row = copy.deepcopy(CHECKPOINTS.get(identity))
    if row is None:
        raise ValueError("权重不在当前目录中，请刷新权重列表或添加本地路径。")
    if not row["available"]:
        raise ValueError(row["reason"])
    return row


def _check_model_config(row, config):
    if config["task_mode"] != row["task_mode"] or config["spatial_dim"] != row["spatial_dim"]:
        raise ValueError("当前训练权重仅支持红方、二维平面、damage 累计伤害任务。")
    for key, limit in row["limits"].items():
        if config[key] > limit:
            raise ValueError(f"{row['method_label']} 的 {key} 上限为 {limit}，当前为 {config[key]}。")


def _register_checkpoint(payload):
    """Explicitly add a trusted local checkpoint; do not load Python from HTTP."""
    if not isinstance(payload, dict) or set(payload) - {"path", "method"}:
        raise ValueError("请提供 path 和可选的 method。")
    value = payload.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("请输入本地 checkpoint 文件路径。")
    path = Path(value.strip()).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError("路径必须指向已有的 .pt checkpoint 文件。")
    if path.name == "resume.pt":
        raise ValueError("请选择 best.pt、final.pt 或 latest.pt；resume.pt 用于恢复训练。")
    import torch
    with path.open("rb") as source:
        saved = torch.load(source, map_location="cpu", weights_only=False)
        stat = os.fstat(source.fileno())
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict) or not isinstance(saved.get("networks"), dict):
        raise ValueError("权重必须包含 config 和 networks，单独 state_dict 或历史研究模型不能直接加载。")
    row = _describe_checkpoint(path, saved["config"])
    if payload.get("method") and payload["method"] != row["method"]:
        raise ValueError("所选算法与 checkpoint 中的 method 不一致。")
    if not row["available"]:
        raise ValueError(row["reason"])
    row["training_steps"] = saved.get("progress", {}).get("t_env")
    row.update(file_size=stat.st_size, modified_ns=str(stat.st_mtime_ns))
    with CATALOG_LOCK:
        CUSTOM_PATHS.add(path)
        CHECKPOINTS[row["id"]] = row
    return row


def _defaults():
    from open_score.envs import had_config
    from open_score.rules.coverage_rule import validate_strategy
    return {**DEFAULTS, "max_steps": int(had_config.DefaultMaxSteps), "target_health": float(had_config.initial_health),
            "red_strategy": validate_strategy("red"), "blue_strategy": validate_strategy("blue")}


def _json(handler, code, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _meta(refresh=False):
    from open_score.algos import METHODS
    from open_score.envs import PHYSICS_PROTOCOL
    from open_score.envs.features import MAX_AGENTS, MAX_BLUE, MAX_TARGETS
    from open_score.envs.scales import VALIDATION_POOL, TEST_POOL
    from open_score.rules.coverage_rule import (
        STRATEGY_CATALOG, register_end_to_end_policy,
        TASK_MODES, env_constants, parameter_snapshot,
    )
    with CATALOG_LOCK:
        if not CATALOG_READY or refresh:
            _scan_checkpoints()
        if CHECKPOINT_POLICY not in STRATEGY_CATALOG["red"]["end_to_end"]["policies"]:
            register_end_to_end_policy("red", CHECKPOINT_POLICY, "Viewer checkpoint", _checkpoint_factory)
        catalog = copy.deepcopy(STRATEGY_CATALOG)
        catalog["blue"]["end_to_end"]["policies"]["nearest_target"] = "各自冲最近目标"
        for layer in catalog["blue"]["hierarchical"]["layers"]:
            if layer["id"] == "grouping":
                layer["policies"].update(reactive="自适应集火单目标（训练）",
                                         concentrated="集火最近目标", balanced="均分攻击多目标")
        catalog["red"]["end_to_end"]["policies"].pop(CHECKPOINT_POLICY, None)
        catalog["red"]["checkpoint"] = {"label": "训练模型（含分层）"}
        checkpoints = copy.deepcopy(list(CHECKPOINTS.values()))
    constants = env_constants()
    constants["bounds"] = constants["bounds"].tolist()
    return {
        "viewer_api": 5,
        "server_pid": os.getpid(),
        "physics_protocol": PHYSICS_PROTOCOL,
        "strategy_catalog": catalog,
        "checkpoints": checkpoints,
        "methods": {method: METHOD_LABELS.get(method, method) for method in METHODS},
        "limits": dict(red=MAX_AGENTS, blue=MAX_BLUE, targets=MAX_TARGETS),
        "scale_presets": [dict(label=scale.name, red=scale.N_R, blue=scale.N_B, targets=scale.K)
                          for scale in dict.fromkeys((*VALIDATION_POOL, *TEST_POOL))],
        "task_modes": TASK_MODES,
        "target_initializations": {"random": "随机目标（训练默认）", "fixed": "固定目标（对比布局）"},
        "spatial_dims": {"2": "二维平面", "3": "三维空间"},
        "defaults": _defaults(),
        "constants": constants,
        "attack_ranges": {str(dim): parameter_snapshot(dim)["AttackDistance"] for dim in (2, 3)},
        "parameters": parameter_snapshot(),
    }


def _validate_config(payload):
    from open_score.envs import PHYSICS_PROTOCOL
    from open_score.rules.coverage_rule import validate_strategy, TASK_MODES
    from open_score.rules.coverage_rule import parameter_snapshot
    if not isinstance(payload, dict):
        raise ValueError("运行参数必须是 JSON 对象。")
    config = _defaults()
    unknown = set(payload) - set(config)
    if unknown:
        raise ValueError("不支持的配置字段，请刷新页面使用当前策略接口：" + ", ".join(sorted(unknown)))
    limits = {
        "targets": (1, 6, "目标数量 K"),
        "red": (1, 40, "红方数量"),
        "blue": (1, 40, "蓝方数量"),
        "stats": (1, 400, "统计局数 S"),
        "display": (0, 20, "回放局数 D"),
        "seed0": (0, MAX_SAFE_INTEGER, "起始种子"),
        "max_steps": (1, 500, "每局物理步数上限"),
        "spatial_dim": (2, 3, "空间维度"),
    }
    for name, (low, high, label) in limits.items():
        value = payload.get(name, config[name])
        try:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError
            number = int(value)
            if isinstance(value, float) and number != value:
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{label}必须是 {low}–{high} 之间的整数。") from None
        if not low <= number <= high:
            raise ValueError(f"{label}必须是 {low}–{high} 之间的整数。")
        config[name] = number
    if config["display"] > config["stats"]:
        raise ValueError("回放局数 D 不能大于统计局数 S。")
    if config["seed0"] + config["stats"] - 1 > MAX_SAFE_INTEGER:
        raise ValueError("最后一局的种子超出安全整数范围，请减小起始种子。")
    mode = payload.get("task_mode", config["task_mode"])
    if not isinstance(mode, str) or mode not in TASK_MODES:
        raise ValueError("不支持的任务模式。")
    config["task_mode"] = mode
    target_mode = payload.get("target_initialization", config["target_initialization"])
    if not isinstance(target_mode, str) or target_mode not in ("random", "fixed"):
        raise ValueError("目标初始化必须为 random 或 fixed。")
    config["target_initialization"] = target_mode
    for side in ("red", "blue"):
        spec = payload.get(f"{side}_strategy")
        if isinstance(spec, dict) and spec.get("architecture") == "checkpoint":
            if side != "red" or set(spec) != {"architecture", "checkpoint"}:
                raise ValueError("训练模型只支持红方，需提供 architecture 和 checkpoint。")
            row = _checkpoint_row(spec["checkpoint"])
            _check_model_config(row, config)
            config[f"{side}_strategy"] = dict(spec)
        else:
            if isinstance(spec, dict) and spec.get("policy") == CHECKPOINT_POLICY:
                raise ValueError("请通过训练模型入口选择 checkpoint。")
            config[f"{side}_strategy"] = validate_strategy(side, spec)
    try:
        value = payload.get("target_health", config["target_health"])
        if isinstance(value, bool):
            raise ValueError
        config["target_health"] = float(value)
        if not math.isfinite(config["target_health"]) or config["target_health"] <= 0:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError("目标初始血量必须是有限正数。") from None
    config["physics_protocol"] = PHYSICS_PROTOCOL
    config["environment_parameters"] = parameter_snapshot(config["spatial_dim"])
    return config


def _progress(record):
    """Copy a job's lightweight state while holding LOCK."""
    payload = {key: value for key, value in record.items() if key != "episodes"}
    payload["config"] = dict(record["config"])
    payload["recorded"] = list(record["recorded"])
    end = record.get("finished_at", time.time())
    payload["elapsed_seconds"] = max(0.0, end - record["started_at"])
    return payload


def _run_job(job_id, payload):
    with RUN_LOCK:
        try:
            _evaluate_job(job_id, payload)
        finally:
            POLICY_CONTEXT.__dict__.clear()


def _evaluate_job(job_id, payload):
    try:
        from open_score.rules.coverage_rule import evaluate
        strategies = {side: payload[f"{side}_strategy"] for side in ("red", "blue")}
        if strategies["red"]["architecture"] == "checkpoint":
            from open_score.algos import load_policy
            import torch
            torch.set_num_threads(1)
            with LOCK:
                JOBS[job_id]["phase"] = "loading"
            row = _checkpoint_row(strategies["red"]["checkpoint"])
            path = Path(row["path"])
            path = ROOT / path if not path.is_absolute() else path
            # Both reads use the same open file, even if training atomically
            # replaces best/latest in the meantime. Keep one policy per job.
            with path.open("rb") as source:
                saved = torch.load(source, map_location="cpu", weights_only=False)
                if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict) or not isinstance(saved.get("networks"), dict):
                    raise ValueError("checkpoint 缺少 config / networks。")
                snapshot = _describe_checkpoint(path, saved["config"])
                if not snapshot["available"]:
                    raise ValueError(snapshot["reason"])
                _check_model_config(snapshot, payload)
                snapshot["training_steps"] = saved.get("progress", {}).get("t_env")
                stat = os.fstat(source.fileno())
                snapshot.update(file_size=stat.st_size, modified_ns=str(stat.st_mtime_ns))
                del saved
                source.seek(0)
                POLICY_CONTEXT.policy = load_policy(snapshot["method"], source)
            payload = {**payload, "model_snapshots": {"red": snapshot}}
            strategies["red"] = {"architecture": "end_to_end", "policy": CHECKPOINT_POLICY}
        with LOCK:
            JOBS[job_id].update(config=dict(payload), phase="evaluating", evaluation_started_at=time.time())
        seeds = list(range(payload["seed0"], payload["seed0"] + payload["stats"]))

        def on_episode(current, episode, keep):
            for side in ("red", "blue"):
                episode[f"{side}_strategy"] = payload[f"{side}_strategy"]
            with LOCK:
                record = JOBS[job_id]
                elapsed = max(0.0, time.time() - record["started_at"])
                average = (time.time() - record["evaluation_started_at"]) / current
                record.update({
                    "current": current, "elapsed_seconds": elapsed,
                    "seconds_per_episode": average,
                    "eta_seconds": average * (record["total"] - current),
                })
                if keep:
                    record["episodes"].append({key: episode[key] for key in (
                        "seed", "success", "outcome", "steps", "frames", "fires",
                        "target_health", "remaining_red", "remaining_blue",
                        "task_mode", "target_damage", "step_target_damage",
                        "target_damage_by_target", "episode_returns", "terminated", "truncated",
                        "spatial_dim", "plane_altitude",
                        "red_strategy", "blue_strategy", "physics_protocol", "bounds",
                        "attack_distance", "fire_range",
                        "target_initialization",
                    )})
                    record["episodes"][-1]["model_snapshots"] = payload.get("model_snapshots", {})
                    record["recorded"].append({
                        "index": current - 1,
                        **{key: episode[key] for key in (
                            "seed", "outcome", "steps", "success",
                            "task_mode", "target_damage",
                            "spatial_dim",
                        )},
                    })

        result = evaluate(
            seeds,
            targets=payload["targets"],
            red=payload["red"],
            blue=payload["blue"],
            red_strategy=strategies["red"],
            blue_strategy=strategies["blue"],
            task_mode=payload["task_mode"],
            target_health=payload["target_health"],
            max_steps=payload["max_steps"],
            spatial_dim=payload["spatial_dim"],
            target_initialization=payload["target_initialization"],
            record_first=payload["display"],
            on_episode=on_episode,
        )
        summary = {key: result[key] for key in (
            "n", "wins", "win_rate", "wilson", "breached", "horizon", "wiped",
            "engage_rate", "kills_per_fire", "clip_rate", "mean_steps",
            "forced_contact", "outcomes", "breach_steps",
            "task_mode", "mean_target_damage", "mean_red_return", "mean_blue_return",
            "target_damage_by_episode", "red_returns", "blue_returns",
            "mean_damage_by_target", "zero_damage_count",
        )}
        with LOCK:
            finished_at = time.time()
            JOBS[job_id].update({
                "done": True, "phase": "complete", "error": None, "current": len(seeds),
                "summary": summary, "finished_at": finished_at,
                "elapsed_seconds": finished_at - JOBS[job_id]["started_at"],
                "eta_seconds": 0.0,
            })
    except Exception as error:
        with LOCK:
            finished_at = time.time()
            JOBS[job_id].update({
                "done": True, "phase": "failed", "error": f"评估运行失败：{error}",
                "finished_at": finished_at,
                "elapsed_seconds": finished_at - JOBS[job_id]["started_at"],
                "eta_seconds": None,
            })


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in ("/", "/viewer.html"):
            data = HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/meta":
            _json(self, 200, _meta(refresh=query.get("refresh") == ["1"]))
            return
        if parsed.path == "/api/jobs":
            with LOCK:
                jobs = [
                    {"job": job_id, **_progress(record)}
                    for job_id, record in JOBS.items() if not record["done"]
                ]
            jobs.sort(key=lambda job: job["started_at"], reverse=True)
            _json(self, 200, {"jobs": jobs})
            return
        if parsed.path == "/api/progress":
            job = query.get("job", [""])[0]
            with LOCK:
                record = JOBS.get(job)
                payload = _progress(record) if record is not None else None
            _json(self, 200 if payload else 404, payload or {"error": "任务不存在，服务可能已重启。"})
            return
        if parsed.path == "/api/episode":
            job = query.get("job", [""])[0]
            try:
                index = int(query.get("index", ["0"])[0])
            except ValueError:
                _json(self, 400, {"error": "回放索引必须是从 0 开始的整数。"})
                return
            with LOCK:
                record = JOBS.get(job)
                if record is None:
                    code, response = 404, {"error": "任务不存在，服务可能已重启。"}
                elif 0 <= index < len(record["episodes"]):
                    code, response = 200, record["episodes"][index]
                elif index < 0 or index >= record["config"]["display"]:
                    code, response = 404, {"error": "此局未设置记录回放。"}
                elif not record["done"]:
                    code, response = 202, {"error": "此局仍在运行，完成后即可回放。"}
                elif record["error"]:
                    code, response = 500, {"error": record["error"]}
                else:
                    code, response = 404, {"error": "此局没有可用回放。"}
            _json(self, code, response)
            return
        _json(self, 404, {"error": "接口不存在。"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/run", "/api/checkpoints"):
            _json(self, 404, {"error": "接口不存在。"})
            return
        origin = self.headers.get("Origin")
        port = self.server.server_port
        if (origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")) or self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            _json(self, 403, {"error": "请从本地 viewer 页面提交 JSON 请求。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 16_384:
                raise ValueError
        except ValueError:
            _json(self, 400, {"error": "请求长度无效或参数过大。"})
            return
        try:
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            _json(self, 400, {"error": "运行参数必须是有效的 UTF-8 JSON。"})
            return
        if parsed.path == "/api/checkpoints":
            try:
                row = _register_checkpoint(payload)
            except Exception as error:
                _json(self, 400, {"error": f"无法添加 checkpoint：{error}"})
                return
            _json(self, 200, {"checkpoint": row, "meta": _meta()})
            return
        try:
            payload = _validate_config(payload)
        except ValueError as error:
            _json(self, 400, {"error": str(error)})
            return
        job_id = uuid.uuid4().hex
        with LOCK:
            JOBS[job_id] = {
                "done": False, "phase": "queued", "error": None, "current": 0,
                "total": payload["stats"], "summary": None,
                "config": dict(payload), "started_at": time.time(),
                "elapsed_seconds": 0.0, "seconds_per_episode": None,
                "eta_seconds": None, "recorded": [], "episodes": [],
            }
        thread = threading.Thread(target=_run_job, args=(job_id, payload), daemon=True)
        thread.start()
        _json(self, 200, {"job": job_id})


def _bind_server(port):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler, bind_and_activate=False)
    server.allow_reuse_address = False
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        server.server_bind()
        server.server_activate()
    except OSError:
        server.server_close()
        raise
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None,
                        help="指定本地端口；默认从 8765 开始选择空闲端口")
    parser.add_argument("--policy-module", action="append", default=[], metavar="MODULE_OR_FILE",
                        help="导入本地 Python 模块；模块通过 register_end_to_end_policy 注册自定义策略，可重复传入")
    args = parser.parse_args()
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1–65535 之间。")
    for value in args.policy_module:
        path = Path(value)
        if path.suffix == ".py":
            path = path.resolve()
            spec = importlib.util.spec_from_file_location(f"viewer_policy_{uuid.uuid4().hex}", path)
            if spec is None or spec.loader is None:
                raise ValueError(f"无法导入策略文件：{path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            importlib.import_module(value)
    meta = _meta()  # 检查依赖和模型目录后才绑定端口。
    ports = (args.port,) if args.port is not None else range(8765, 8776)
    for port in ports:
        try:
            server = _bind_server(port)
            break
        except OSError as error:
            if error.errno != errno.EADDRINUSE and getattr(error, "winerror", None) != 10048:
                raise
    else:
        parser.exit(1, "端口已被占用。请指定其他端口，例如：python viewer.py --port 8776\n")
    if args.port is None and port != 8765:
        print(f"8765 已被占用（可能仍是旧版 viewer），本次改用 {port}。请打开下方新地址。", flush=True)
    count = sum(row["available"] for row in meta["checkpoints"])
    print(f"Viewer API {meta['viewer_api']} · PID {os.getpid()} · {count} 个可用模型", flush=True)
    print(f"Open http://127.0.0.1:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
