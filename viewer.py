"""Local HAD viewer with symmetric end-to-end / hierarchical strategy inputs."""
from __future__ import annotations

import json
import math
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "viewer.html"
JOBS = {}
LOCK = threading.Lock()
DEFAULTS = {
    "targets": 2, "red": 8, "blue": 8, "seed0": 0,
    "display": 5, "stats": 100,
    "task_mode": "damage",
    "spatial_dim": 2,
    "target_initialization": "random",
}
MAX_SAFE_INTEGER = 2 ** 53 - 1


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


def _meta():
    from open_score.envs import PHYSICS_PROTOCOL
    from open_score.rules.coverage_rule import (
        STRATEGY_CATALOG,
        TASK_MODES, env_constants, parameter_snapshot,
    )
    constants = env_constants()
    constants["bounds"] = constants["bounds"].tolist()
    return {
        "viewer_api": 4,
        "physics_protocol": PHYSICS_PROTOCOL,
        "strategy_catalog": STRATEGY_CATALOG,
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
        "targets": (1, 3, "目标数量 K"),
        "red": (1, 32, "红方数量"),
        "blue": (1, 32, "蓝方数量"),
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
        config[f"{side}_strategy"] = validate_strategy(side, payload.get(f"{side}_strategy"))
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
    try:
        from open_score.rules.coverage_rule import evaluate
        seeds = list(range(payload["seed0"], payload["seed0"] + payload["stats"]))

        def on_episode(current, episode, keep):
            with LOCK:
                record = JOBS[job_id]
                elapsed = max(0.0, time.time() - record["started_at"])
                average = elapsed / current
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
            red_strategy=payload["red_strategy"],
            blue_strategy=payload["blue_strategy"],
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
                "done": True, "error": None, "current": len(seeds),
                "summary": summary, "finished_at": finished_at,
                "elapsed_seconds": finished_at - JOBS[job_id]["started_at"],
                "eta_seconds": 0.0,
            })
    except Exception as error:
        with LOCK:
            finished_at = time.time()
            JOBS[job_id].update({
                "done": True, "error": f"评估运行失败：{error}",
                "finished_at": finished_at,
                "elapsed_seconds": finished_at - JOBS[job_id]["started_at"],
                "eta_seconds": None,
            })


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
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
            _json(self, 200, _meta())
            return
        query = parse_qs(parsed.query)
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
        if parsed.path != "/api/run":
            _json(self, 404, {"error": "接口不存在。"})
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
        try:
            payload = _validate_config(payload)
        except ValueError as error:
            _json(self, 400, {"error": str(error)})
            return
        job_id = uuid.uuid4().hex
        with LOCK:
            JOBS[job_id] = {
                "done": False, "error": None, "current": 0,
                "total": payload["stats"], "summary": None,
                "config": dict(payload), "started_at": time.time(),
                "elapsed_seconds": 0.0, "seconds_per_episode": None,
                "eta_seconds": None, "recorded": [], "episodes": [],
            }
        thread = threading.Thread(target=_run_job, args=(job_id, payload), daemon=True)
        thread.start()
        _json(self, 200, {"job": job_id})


def main():
    _meta()  # 先检查环境依赖；避免未安装 had_env 的解释器占用端口后才报错。
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler, bind_and_activate=False)
    server.allow_reuse_address = False
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    server.server_bind()
    server.server_activate()
    print("Open http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
