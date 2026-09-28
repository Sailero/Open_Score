"""Scientific figures and deterministic videos from recorded physical states."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .playback import display_frame

COLORS = {"red": "#ef6b73", "blue": "#55a7ed", "targets": "#dba940"}
MARKERS = {"Attack": "o", "Scout": "^", "Disturb": "D", "Entity": "P"}


class EpisodeFigure:
    def __init__(self, episode, *, light=True, width=1600, height=1000):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self.episode, self.light = episode, light
        self.figure = Figure(figsize=(width / 100, height / 100), dpi=100)
        self.canvas = FigureCanvasAgg(self.figure)
        grid = self.figure.add_gridspec(2, 2, width_ratios=(3, 1), height_ratios=(4, 1),
                                      left=.065, right=.975, bottom=.08, top=.9, hspace=.3, wspace=.3)
        self.map = self.figure.add_subplot(grid[0, 0])
        self.altitude = self.figure.add_subplot(grid[1, 0])
        self.metrics = self.figure.add_subplot(grid[0, 1])
        self.caption = self.figure.add_subplot(grid[1, 1])
        self.background = "#ffffff" if light else "#0f172a"
        self.ink = "#243447" if light else "#e2e8f0"
        self.figure.set_facecolor(self.background)
        self.title = self.figure.suptitle("", fontsize=15, color=self.ink, x=.065, ha="left")

    def draw(self, frame):
        for ax in (self.map, self.altitude, self.metrics, self.caption):
            ax.clear()
            ax.set_facecolor(self.background)
            ax.tick_params(colors=self.ink, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#cbd5e1" if self.light else "#334155")
            ax.xaxis.label.set_color(self.ink)
            ax.yaxis.label.set_color(self.ink)
        step = frame["step"]
        meta = self.episode.metadata
        protocol = meta.get("protocol_id", meta.get("protocol", "HAD"))
        label = "SPARSE / recorded samples only" if self.episode.sparse else "recorded physical states"
        self.title.set_text(f"HAD  |  {protocol}  |  step {step}/{self.episode.max_step}  |  {label}")
        def located(e):
            return isinstance(e.get("position"), (list, tuple)) and len(e["position"]) == 3 and all(
                isinstance(v, (float, int)) and math.isfinite(v) for v in e["position"])
        entities = {(e["side"], e["id"]): e for e in frame["entities"] if located(e)}
        trajectories = {}
        for old in self.episode.frames:
            if old["step"] > step:
                break
            for e in old["entities"]:
                if located(e):
                    trajectories.setdefault((e["side"], e["id"]), []).append(e["position"])
        for key, entity in entities.items():
            side = entity["side"]
            color = COLORS.get(side, "#94a3b8") if entity.get("alive", True) else "#94a3b8"
            x, y, z = entity["position"]
            points = np.asarray(trajectories.get(key, []))
            if len(points) > 1 and side != "targets":
                self.map.plot(points[:, 0], points[:, 1], color=color, alpha=.27, linewidth=1)
            self.map.scatter([x], [y], c=color, s=85 if side == "targets" else 35,
                             marker=MARKERS.get(entity.get("role"), "o"), zorder=4)
            if entity.get("alive", True) and side != "targets":
                vx, vy, _ = entity.get("velocity") or (0, 0, 0)
                self.map.quiver(x, y, vx, vy, angles="xy", scale_units="xy", scale=1,
                                color=color, width=.0025, zorder=3)
            if len(entities) <= 26 or side == "targets":
                self.map.annotate(f"{side[0].upper()}{entity['id']}", (x, y),
                                  xytext=(5, 5), textcoords="offset points", fontsize=8, color=self.ink)
            self.altitude.scatter([x], [z], c=color, s=24)
        self.map.set(xlim=(-2500, 2500), ylim=(-2500, 2500), xlabel="x (m)", ylabel="y (m)")
        self.map.set_aspect("equal", adjustable="box")
        self.altitude.set(xlim=(-2500, 2500), ylim=(0, 1000), xlabel="x (m)", ylabel="height (m)")
        for ax in (self.map, self.altitude):
            ax.grid(alpha=.15)
        for side in ("red", "blue"):
            counts = [sum(e.get("alive", True) and e["side"] == side for e in f["entities"])
                      for f in self.episode.frames]
            self.metrics.step([f["step"] for f in self.episode.frames], counts,
                              where="post", label=side.title(), color=COLORS[side], linewidth=1.6)
        self.metrics.axvline(step, color=self.ink, alpha=.4, linestyle="--")
        self.metrics.set(xlabel="physical step", ylabel="surviving agents")
        self.metrics.legend(frameon=False, labelcolor=self.ink)
        self.metrics.grid(alpha=.15)
        self.caption.axis("off")
        scenario = meta.get("scenario", {})
        seed = scenario.get("opening_seed", scenario.get("seed", meta.get("seed", "unknown")))
        target_hp = ", ".join(f"T{e['id']}: " + (f"{e['health']:.2f}" if isinstance(e.get("health"), (int, float)) else "unknown")
                              for e in frame["entities"] if e["side"] == "targets")
        self.caption.text(0, 1, f"Seed: {seed}\nTarget HP: {target_hp}\nVelocity arrows: 1 s displacement\nHeight projection: XZ",
                          va="top", color=self.ink, fontsize=10, linespacing=1.65)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self):
        self.figure.clear()


def export_figure(episode, path, *, step=None, light=True, width=1600, height=1000):
    path = Path(path)
    if path.suffix.lower() not in (".png", ".svg", ".pdf"):
        raise ValueError("Figure format must be PNG, SVG or PDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = EpisodeFigure(episode, light=light, width=width, height=height)
    try:
        figure.draw(episode.frame_at(episode.max_step if step is None else step))
        figure.figure.savefig(path, facecolor=figure.background)
    finally:
        figure.close()
    _sidecar(episode, path)
    return path


def export_video(episode, path, *, fps=30, steps_per_second=5., light=False,
                 width=1280, height=800, progress=None):
    import imageio.v2 as imageio
    if fps <= 0 or steps_per_second <= 0:
        raise ValueError("fps and steps_per_second must be positive")
    path = Path(path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("Video format must be MP4")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = EpisodeFigure(episode, light=light, width=width, height=height)
    start_step = episode.frames[0]["step"] if episode.frames else 0
    count = max(1, int(np.ceil((episode.max_step - start_step) / steps_per_second * fps)))
    try:
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_log_level="error") as writer:
            for index in range(count + 1):
                position = min(episode.max_step, start_step + index / fps * steps_per_second)
                writer.append_data(figure.draw(display_frame(episode, position)))
                if progress:
                    progress(index + 1, count + 1)
    finally:
        figure.close()
    _sidecar(episode, path, fps=fps, steps_per_second=steps_per_second)
    return path


def _sidecar(episode, path, **extra):
    payload = {"schema": "had-export-v1", "recording_metadata": episode.metadata,
               "sparse": episode.sparse, **extra}
    Path(str(path) + ".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
