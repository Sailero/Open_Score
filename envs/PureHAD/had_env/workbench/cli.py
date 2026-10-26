"""Command-line entry points for isolated HAD research and replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys

def _output_root(module_file=None):
    root = Path(module_file or __file__).resolve().parents[2]
    # Editable/source checkouts keep their outputs together. Wheels must never
    # assume site-packages is a writable user-data directory.
    return root if (root / "pyproject.toml").is_file() else Path.cwd()


WORKBENCH_ROOT = _output_root()


def parser():
    root = argparse.ArgumentParser(prog="had-workbench", description="HAD 科研工作台：独立调试、记录、回放与对比")
    subs = root.add_subparsers(dest="command")
    for name in ("record", "live"):
        p = subs.add_parser(name, help="录制完整回合" if name == "record" else "启动独立交互调试会话")
        p.add_argument("--red", type=int, default=8)
        p.add_argument("--blue", type=int, default=8)
        p.add_argument("--seed", type=int, default=20260907)
        p.add_argument("--opponent-seed", type=int, default=20260908)
        p.add_argument("--policy", default="rule", choices=("rule", "grand", "static_rule", "random"))
        p.add_argument("--scenario", type=Path, help="ScenarioSpec JSON")
        p.add_argument("--native", action="store_true", help="原生双方飞行协议；默认固定对手动态分组")
        p.add_argument("--scouts", type=int, default=0, help="原生协议下每方侦察机数量")
        p.add_argument("--disturbers", type=int, default=0, help="原生协议下每方干扰机数量")
        p.add_argument("--output", type=Path,
                       default=WORKBENCH_ROOT / "outputs/workbench" / f"episode_{datetime.now():%Y%m%d_%H%M%S_%f}.json.gz")
    p = subs.add_parser("view", help="查看录制或历史 v5/v4 轨迹")
    p.add_argument("episodes", nargs="*", type=Path)
    p.add_argument("--compare", type=Path)
    p.add_argument("--watch", type=Path, help="只读观察录制目录或研究输出目录")
    p = subs.add_parser("export", help="导出科研图或视频")
    p.add_argument("episode", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--step", type=int)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--steps-per-second", type=float, default=5.)
    p = subs.add_parser("branch", help="在录制的上层决策边界重建并创建独立分支")
    p.add_argument("episode", type=Path)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--policy", choices=("rule", "grand", "static_rule", "random"), default="grand")
    p.add_argument("--output", required=True, type=Path)
    p = subs.add_parser("evaluate", help="按显式 ID/OOD 场景清单进行配对评估")
    p.add_argument("plan", type=Path)
    p.add_argument("--policies", nargs="+", default=["rule", "grand"])
    p.add_argument("--training-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--output", type=Path, default=WORKBENCH_ROOT / "outputs/workbench/evaluation")
    p = subs.add_parser("inspect", help="打印回放摘要，无需 Qt")
    p.add_argument("episode", type=Path)
    return root


def _scenario(args):
    if args.native and args.policy != "rule":
        raise ValueError("--policy 用于上层分组协议；原生飞行自定义策略请通过 FlightSession 接入")
    if args.scenario:
        return json.loads(args.scenario.read_text(encoding="utf-8-sig"))
    if args.native:
        from .agent_session import FlightScenarioSpec
        return asdict(FlightScenarioSpec(red_attackers=args.red, blue_attackers=args.blue, seed=args.seed,
                                        red_scouts=args.scouts, blue_scouts=args.scouts,
                                        red_disturbers=args.disturbers, blue_disturbers=args.disturbers))
    if args.scouts or args.disturbers:
        raise ValueError("异构编队使用 --native；固定分组协议的角色不变")
    from .protocols import ScenarioSpec
    return ScenarioSpec(red_count=args.red, blue_count=args.blue, opening_seed=args.seed,
                        opponent_seed=args.opponent_seed).to_dict()


def reconstruct_branch(episode, step, *, policy="grand", seed=20260909):
    from .identity import assert_behavior_compatible
    from .session import PolicyAdapter, SimulationSession
    if episode.sparse:
        raise ValueError("历史稀疏轨迹仅供查看；精确分支需要当前版本完整录制")
    if str(episode.metadata.get("protocol_id", "")).startswith("had-native-flight"):
        raise ValueError("原生飞行分支请使用 FlightSession.branch；CLI此入口为上层决策分支")
    assert_behavior_compatible(episode.metadata.get("source_identity", {}))
    if step not in {d["step"] for d in episode.decisions}:
        raise ValueError("分支位置必须是已记录的上层决策边界")
    selected = next(d for d in episode.decisions if d["step"] == step)
    if selected.get("snapshot"):
        child = SimulationSession.from_snapshot(selected["snapshot"], episode.metadata["scenario"],
                                                policy=policy, continuation_seed=seed, policy_seed=seed)
        try:
            return child.run()
        finally:
            child.close()
    session = SimulationSession(episode.metadata["scenario"])
    try:
        for decision in episode.decisions:
            if decision["step"] >= step:
                break
            session.step(decision["selected_plan"])
        from .recording import capture_frame
        observed = capture_frame(session.env.adapter)["entities"]
        if observed != episode.frame_at(step)["entities"]:
            raise ValueError("重建状态与记录不一致，未创建分支")
        child = session.branch(continuation_seed=seed)
        try:
            child.policy = PolicyAdapter(policy, seed=seed)
            child.episode.metadata["policies"]["red"] = policy
            return child.run()
        finally:
            child.close()
    finally:
        session.close()


def _branch(args):
    from .recording import load_episode
    episode = reconstruct_branch(load_episode(args.episode), args.step, policy=args.policy, seed=args.seed)
    return episode.save(args.output)


def _show(args):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import (QApplication, QMessageBox, QDialog, QFormLayout,
                                  QComboBox, QSpinBox, QDialogButtonBox)
    from .gui import WorkbenchWindow
    from .recording import ReplayEpisode, load_episode
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("HAD Research Workbench")
    window = WorkbenchWindow()
    controller = None
    timers = []
    branch_jobs = []

    def saved_branch(action):
        if action != "branch" or window.live_mode or not window.episode:
            return
        steps = sorted({d["step"] for d in window.episode.decisions if d.get("snapshot")})
        if not steps:
            QMessageBox.information(window, "创建分支", "此录制未保存可恢复决策快照。")
            return
        dialog = QDialog(window)
        dialog.setWindowTitle("从决策快照创建独立分支")
        form = QFormLayout(dialog)
        step_box, policy_box, seed_box = QComboBox(), QComboBox(), QSpinBox()
        step_box.addItems([str(s) for s in steps])
        step_box.setCurrentText(str(max((s for s in steps if s <= window.position), default=steps[0])))
        policy_box.addItems(["grand", "rule", "static_rule", "random"])
        seed_box.setRange(0, 2**31 - 1)
        seed_box.setValue(20260909)
        form.addRow("决策物理步", step_box)
        form.addRow("后续策略", policy_box)
        form.addRow("独立续跑种子", seed_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        from .live import BranchController
        job = BranchController(window.episode, int(step_box.currentText()),
                               policy=policy_box.currentText(), seed=seed_box.value())
        branch_jobs.append(job)
        window.statusBar().showMessage("正在独立进程中续跑分支，原轨迹保持不变。")
        timer = QTimer(window)
        def branch_poll():
            for row in job.poll():
                if row["kind"] == "branch":
                    value = row["episode"]
                    episode = ReplayEpisode(value["metadata"], value["frames"], value["decisions"])
                    target = WORKBENCH_ROOT / "outputs/workbench" / f"branch_{datetime.now():%Y%m%d_%H%M%S_%f}.json.gz"
                    episode.save(target)
                    window.add_episode(episode, f"分支 / {target.stem}")
                    window.set_comparison(episode)
                    window.statusBar().showMessage(f"分支已完成并保存到 {target}")
                    timer.stop()
                    job.close()
                elif row["kind"] == "error":
                    timer.stop()
                    QMessageBox.critical(window, "分支未创建", row["message"])
                    job.close()
            if not job.process.is_alive():
                timer.stop()
        timer.setInterval(50)
        timer.timeout.connect(branch_poll)
        timer.start()
        timers.append(timer)
    window.control_requested.connect(saved_branch)
    app.aboutToQuit.connect(lambda: [job.close() for job in branch_jobs])
    if args.command == "live":
        from .live import LiveController
        controller = LiveController(_scenario(args), args.policy, native=args.native, save_path=str(args.output))
        window.set_live_mode(True)
        def control(action):
            if action == "branch" and not window.live_mode:
                return
            if action.startswith("speed:"):
                controller.send("speed", value=float(action.split(":", 1)[1]))
            else:
                controller.send(action)
        window.control_requested.connect(control)
        def poll():
            for row in controller.poll():
                kind = row["kind"]
                if kind in ("episode", "update", "complete", "branch"):
                    value = row["episode"]
                    episode = ReplayEpisode(value["metadata"], value["frames"], value["decisions"], value.get("sparse", False))
                    if kind == "branch":
                        target = WORKBENCH_ROOT / "outputs/workbench" / f"live_branch_{datetime.now():%Y%m%d_%H%M%S_%f}.json.gz"
                        episode.save(target)
                        window.add_episode(episode, f"实时分支 / {target.stem}")
                        window.set_comparison(episode)
                        window.statusBar().showMessage(f"独立分支已保存到 {target}")
                    else:
                        window.append_episode(episode)
                        window.set_live_mode(kind != "complete")
                    if kind == "complete":
                        window.statusBar().showMessage(f"回合完成，已保存到 {args.output}")
                elif kind == "frame":
                    window.append_frame(row["frame"])
                elif kind == "notice":
                    window.statusBar().showMessage(row["message"], 8000)
                elif kind == "paused":
                    window.pause()
                elif kind == "error":
                    window.set_live_mode(False)
                    QMessageBox.critical(window, "独立调试会话错误", row["message"])
            if not controller.process.is_alive():
                timer.stop()
        timer = QTimer(window)
        timer.setInterval(33)
        timer.timeout.connect(poll)
        timer.start()
        timers.append(timer)
        app.aboutToQuit.connect(controller.close)
    else:
        for path in args.episodes:
            window.add_episode(path)
        if args.episodes:
            window.set_episode(load_episode(args.episodes[0]))
        if args.compare:
            window.set_comparison(load_episode(args.compare))
        if args.watch:
            seen = set()
            def discover():
                if not args.watch.is_dir():
                    return
                # Readers only: no monitor/checkpoint/runner commands are invoked.
                paths = list(args.watch.rglob("*.jsonl.gz")) + list(args.watch.rglob("*.json.gz"))
                paths += list(args.watch.glob("*.json"))
                for path in sorted(set(paths)):
                    stamp = (str(path), path.stat().st_mtime_ns)
                    if stamp in seen:
                        continue
                    seen.add(stamp)
                    try:
                        window.add_episode(path)
                    except (ValueError, KeyError, OSError, json.JSONDecodeError):
                        continue
            discover()
            timer = QTimer(window)
            timer.setInterval(2000)
            timer.timeout.connect(discover)
            timer.start()
            timers.append(timer)
    window.show()
    return app.exec()


def main(argv=None):
    # CLI viewers and rule previews are CPU tasks in their own process.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = parser().parse_args(argv)
    if args.command is None:
        args = parser().parse_args(["live"])
    if args.command in ("view", "live"):
        return _show(args)
    if args.command == "record":
        if args.native:
            from .agent_session import FlightSession
            session = FlightSession(_scenario(args))
        else:
            from .session import SimulationSession
            session = SimulationSession(_scenario(args), args.policy)
        try:
            session.run()
            session.recorder.save(args.output)
            meta = session.recorder.episode.metadata
            print(json.dumps({"output": str(args.output), "frames": len(session.recorder.episode.frames),
                              "outcome": meta.get("outcome_red"), "reason": meta.get("termination_reason"),
                              "protocol": meta.get("protocol_id")}, ensure_ascii=False, default=str))
        finally:
            session.close()
    elif args.command == "export":
        from .recording import load_episode
        from .export import export_figure, export_video
        episode = load_episode(args.episode)
        if args.output.suffix.lower() == ".mp4":
            export_video(episode, args.output, fps=args.fps, steps_per_second=args.steps_per_second, light=not args.dark)
        else:
            export_figure(episode, args.output, step=args.step, light=not args.dark)
        print(args.output)
    elif args.command == "branch":
        print(_branch(args))
    elif args.command == "inspect":
        from .recording import load_episode
        episode = load_episode(args.episode)
        print(json.dumps(dict(metadata=episode.metadata, frames=len(episode.frames),
                              decisions=len(episode.decisions), decision_steps=[d["step"] for d in episode.decisions], max_step=episode.max_step,
                              sparse=episode.sparse), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        from .evaluation import EvaluationPlan, evaluate_policies, summarize_evaluation
        plan = EvaluationPlan.from_dict(json.loads(args.plan.read_text(encoding="utf-8-sig")))
        args.output.mkdir(parents=True, exist_ok=True)
        result = evaluate_policies({name: name for name in args.policies}, plan.test,
                                   training_seeds=args.training_seeds, record_dir=args.output / "episodes")
        for name, data in (("plan.json", plan.to_dict()), ("episodes.json", result["episodes"]),
                           ("summary.json", {k: v for k, v in result.items() if k != "episodes"})):
            (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(args.output / "summary.json")
    return 0
