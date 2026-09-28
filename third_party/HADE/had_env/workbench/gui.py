"""HAD desktop research workbench. Rendering never advances simulation state."""
from __future__ import annotations

import bisect
import json
import math
import time
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QKeySequence, QPainter, QPageLayout, QPageSize, QPdfWriter
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSlider, QSplitter, QStyle, QStyleOptionSlider,
    QTabWidget, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .recording import ReplayEpisode, load_episode
from .views import BattlefieldView, entity_key, ensure_fonts


DARK_STYLE = """
QMainWindow, QWidget { background: #111c2d; color: #dce6f3; font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 12px; }
QToolBar { background: #17243a; border: none; spacing: 8px; padding: 8px; }
QDockWidget { font-weight: 600; }
QDockWidget::title { background: #1a2940; padding: 8px; }
QLineEdit, QComboBox, QPlainTextEdit, QTreeWidget, QListWidget { background: #0d1727; border: 1px solid #2c3c54; border-radius: 5px; padding: 5px; }
QPushButton { background: #243854; border: 1px solid #3b5272; border-radius: 5px; padding: 6px 10px; }
QPushButton:hover { background: #324c70; }
QPushButton:disabled { color: #64738a; }
QListWidget::item { padding: 9px 4px; border-bottom: 1px solid #213149; }
QListWidget::item:selected, QTreeWidget::item:selected { background: #284568; }
QHeaderView::section { background: #213149; padding: 5px; border: 0; }
QSlider::groove:horizontal { background: #2e405a; height: 5px; border-radius: 2px; }
QSlider::handle:horizontal { background: #a8ccff; width: 13px; margin: -5px 0; border-radius: 6px; }
QStatusBar { background: #17243a; color: #a8b8cc; }
QSplitter::handle { background: #2c3c54; }
"""

EVENT_NAMES = {"fire": "开火", "attack_damage": "攻击伤害", "damage": "受损", "death": "死亡",
               "collision": "碰撞", "self_destruct": "自毁", "agents_destroyed": "单位损毁",
               "target_destroyed": "目标损毁", "targets_breached": "目标失守", "red_win": "红方胜",
               "blue_win": "蓝方胜", "terminal": "终局", "horizon": "达到时限"}


class EventSlider(QSlider):
    """Physical-step slider with true decision and event markers."""

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.markers: list[tuple[int, str]] = []
        self.setMinimumHeight(34)

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self)
        span = max(1, self.width() - handle.width())
        painter = QPainter(self)
        for step, kind in self.markers:
            x = handle.width()/2 + QStyle.sliderPositionFromValue(self.minimum(), self.maximum(), step, span)
            painter.setPen(QColor({"event": "#ff797f", "decision": "#f2cd77", "divergence": "#bc94ff"}[kind]))
            painter.drawLine(QPointF(x, self.height()-8), QPointF(x, self.height()-3))
            if kind == "divergence":
                painter.drawLine(QPointF(x, 2), QPointF(x, self.height()-3))
        painter.end()


class WorkbenchWindow(QMainWindow):
    """Offline viewer, or a front end to an independently owned live worker.

    Live callbacks: ``control_requested`` emits play/pause/step/branch. A host
    may deliver frames via ``append_frame`` on the GUI thread. Closing this
    window only stops its display timer; the host owns worker lifecycle.
    """

    control_requested = Signal(str)
    frame_changed = Signal(int)

    def __init__(self, episode: ReplayEpisode | None = None, parent=None):
        super().__init__(parent)
        ensure_fonts()
        self.setWindowTitle("HAD · 攻防研究工作台")
        self.resize(1580, 1020)
        self.episode: ReplayEpisode | None = None
        self.comparison: ReplayEpisode | None = None
        self.position = 0.0
        self.playing = False
        self.live_mode = False
        self.follow_live = True
        self._last_tick = time.perf_counter()
        self._episodes: list[tuple[str, ReplayEpisode]] = []
        self._steps: list[int] = []
        self._decision_steps: list[int] = []
        self._selected: tuple[str, str] | None = None
        self._inspector_comparison = False
        self._frame: dict = {}
        self._comparison_divergence: int | None = None
        self._decision_cache = None
        self._building = False
        self.setStyleSheet(DARK_STYLE)
        self._build_toolbar()
        self._build_canvas()
        self._build_library()
        self._build_inspector()
        self._build_timeline()
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.statusBar().showMessage("打开记录文件开始回放 · 支持 HAD JSON / JSON.GZ 与历史逐步记录")
        if episode is not None:
            self.set_episode(episode)

    def _build_toolbar(self):
        bar = QToolBar("实验与视图", self)
        bar.setMovable(False)
        self.addToolBar(bar)
        title = QLabel("HAD  /  RESEARCH")
        title.setStyleSheet("font-size: 17px; font-weight: 700; color: #a8ccff; padding-right: 14px")
        bar.addWidget(title)
        for label, callback, shortcut in (
            ("打开回合", self._open_dialog, QKeySequence.StandardKey.Open),
            ("并排对比", self._compare_dialog, None),
            ("导出画面", self._export_dialog, None),
            ("适应战场", self._fit_views, None),
        ):
            action = QAction(label, self)
            action.triggered.connect(callback)
            if shortcut is not None:
                action.setShortcut(shortcut)
            bar.addAction(action)
        bar.addSeparator()
        self.layer_actions = {}
        for key, text in (("labels", "标签"), ("trails", "轨迹"), ("groups", "编组"), ("assignments", "任务"), ("ranges", "范围")):
            action = QAction(text, self)
            action.setCheckable(True)
            action.setChecked(key in {"trails", "groups"})
            if key == "trails":
                action.setToolTip("最近 12 个真实采样帧的短轨迹；完整轨迹可导出科研图")
            action.toggled.connect(lambda checked, name=key: self.set_layer(name, checked))
            bar.addAction(action)
            self.layer_actions[key] = action
        self.light_action = QAction("浅色", self)
        self.light_action.setCheckable(True)
        self.light_action.toggled.connect(self._set_light)
        bar.addAction(self.light_action)
        self.addToolBarBreak()
        info = QToolBar("回合信息", self)
        info.setMovable(False)
        self.addToolBar(info)
        self.metadata_label = QLabel("尚未载入回合")
        self.metadata_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.addWidget(self.metadata_label)

    def _build_canvas(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.comparison_label = QLabel("")
        self.comparison_label.setStyleSheet("padding: 7px 14px; color: #f2cd77;")
        self.comparison_label.hide()
        layout.addWidget(self.comparison_label)
        self.canvas_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.xy_view = BattlefieldView("xy")
        self.compare_view = BattlefieldView("xy")
        self.compare_view.hide()
        self.canvas_splitter.addWidget(self.xy_view)
        self.canvas_splitter.addWidget(self.compare_view)
        layout.addWidget(self.canvas_splitter)
        self.xy_view.entity_selected.connect(self.select_entity)
        self.compare_view.entity_selected.connect(self._inspect_comparison)
        self.setCentralWidget(widget)

    def _dock(self, title, widget, area, name):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, dock)
        return dock

    def _build_library(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        self.filter_text = QLineEdit()
        self.filter_text.setPlaceholderText("搜索算法 / 场景 / 种子 / 终局原因")
        self.filter_text.textChanged.connect(self._filter_library)
        layout.addWidget(self.filter_text)
        filters = QHBoxLayout()
        self.outcome_filter = QComboBox()
        self.outcome_filter.addItems(["全部结果", "红方胜", "蓝方胜", "未决 / 平局", "采样截断", "累计伤害"])
        self.outcome_filter.currentIndexChanged.connect(self._filter_library)
        filters.addWidget(self.outcome_filter)
        self.size_filter = QComboBox()
        self.size_filter.addItem("全部规模")
        self.size_filter.currentIndexChanged.connect(self._filter_library)
        filters.addWidget(self.size_filter)
        layout.addLayout(filters)
        self.episode_list = QListWidget()
        self.episode_list.currentRowChanged.connect(self._library_selected)
        layout.addWidget(self.episode_list)
        open_folder = QPushButton("载入记录目录")
        open_folder.clicked.connect(self._open_directory)
        layout.addWidget(open_folder)
        legend = QLabel("▲ Attack   ● Scout   ◆ Disturb\n■ Target   红 / 蓝为阵营\n金色时间标记：决策   红色：事件")
        legend.setStyleSheet("font-size: 11px; color: #91a4c0; padding: 7px 0")
        legend.setWordWrap(True)
        layout.addWidget(legend)
        dock = self._dock("回合库", widget, Qt.DockWidgetArea.LeftDockWidgetArea, "episode-library")
        dock.setMinimumWidth(250)

    def _build_inspector(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        self.inspector_title = QLabel("点击战场实体查看状态")
        layout.addWidget(self.inspector_title)
        self.entity_tree = QTreeWidget()
        self.entity_tree.setHeaderLabels(["属性", "记录值"])
        self.entity_tree.setRootIsDecorated(False)
        self.entity_tree.setColumnWidth(0, 90)
        layout.addWidget(self.entity_tree, 2)
        layout.addWidget(QLabel("最近上层决策 / 算法解释"))
        self.decision_tabs = QTabWidget()
        self.decision_summary = QPlainTextEdit()
        self.decision_summary.setReadOnly(True)
        self.decision_tabs.addTab(self.decision_summary, "决策摘要")
        self.decision_text = QPlainTextEdit()
        self.decision_text.setReadOnly(True)
        self.decision_text.setPlaceholderText("该策略未提供解释数据")
        self.decision_tabs.addTab(self.decision_text, "原始记录")
        self.decision_tabs.currentChanged.connect(lambda _: self._show_raw_decision())
        layout.addWidget(self.decision_tabs, 2)
        self.branch_button = QPushButton("从当前状态创建分支")
        self.branch_button.setEnabled(False)
        self.branch_button.setToolTip("需要含可恢复快照的专用调试会话")
        self.branch_button.clicked.connect(lambda: self.control_requested.emit("branch"))
        layout.addWidget(self.branch_button)
        dock = self._dock("实体与决策", widget, Qt.DockWidgetArea.RightDockWidgetArea, "entity-inspector")
        dock.setMinimumWidth(290)

    def _build_timeline(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.height_view = BattlefieldView("xz")
        self.height_view.entity_selected.connect(self.select_entity)
        self.height_view.setMinimumHeight(135)
        splitter.addWidget(self.height_view)
        self.event_list = QListWidget()
        self.event_list.itemClicked.connect(self._event_selected)
        self.event_list.setMinimumWidth(340)
        splitter.addWidget(self.event_list)
        splitter.setSizes([850, 400])
        layout.addWidget(splitter)
        controls = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)
        self.step_button = QPushButton("单步 →")
        self.step_button.clicked.connect(self.step_forward)
        controls.addWidget(self.step_button)
        for label, callback in (("下次决策", self.next_decision), ("下次伤亡", self.next_event)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            controls.addWidget(button)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
        self.speed_combo.setCurrentText("1×")
        self.speed_combo.currentTextChanged.connect(self._speed_changed)
        controls.addWidget(self.speed_combo)
        self.interpolate_check = QCheckBox("平滑显示")
        self.interpolate_check.setChecked(True)
        self.interpolate_check.setToolTip("仅插值位姿；暂停、单步显示真实物理帧")
        controls.addWidget(self.interpolate_check)
        self.follow_check = QCheckBox("跟随实时")
        self.follow_check.setChecked(True)
        self.follow_check.toggled.connect(self._set_follow)
        self.follow_check.hide()
        controls.addWidget(self.follow_check)
        controls.addStretch(1)
        self.step_label = QLabel("STEP —  /  TIME —")
        controls.addWidget(self.step_label)
        layout.addLayout(controls)
        self.timeline = EventSlider()
        self.timeline.setRange(0, 0)
        self.timeline.sliderPressed.connect(self.pause)
        self.timeline.valueChanged.connect(self.seek)
        layout.addWidget(self.timeline)
        dock = self._dock("高度、事件与物理时间轴", widget, Qt.DockWidgetArea.BottomDockWidgetArea, "timeline")
        dock.setMinimumHeight(255)
        self.resizeDocks([dock], [290], Qt.Orientation.Vertical)
        action = QAction("播放 / 暂停", self)
        action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        action.triggered.connect(self.toggle_play)
        self.addAction(action)
        action = QAction("下一物理帧", self)
        action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        action.triggered.connect(self.step_forward)
        self.addAction(action)

    @property
    def views(self):
        return self.xy_view, self.height_view, self.compare_view

    def load_episode(self, path):
        episode = load_episode(path)
        self.add_episode(episode, str(Path(path).name))
        self.set_episode(episode)
        return episode

    def add_episode(self, episode_or_path, name: str | None = None):
        if isinstance(episode_or_path, (str, Path)):
            name = name or Path(episode_or_path).name
            episode_or_path = load_episode(episode_or_path)
        episode = episode_or_path
        name = name or str(episode.metadata.get("episode_id", f"回合 {len(self._episodes)+1}"))
        index = len(self._episodes)
        self._episodes.append((name, episode))
        item = QListWidgetItem(self._episode_caption(name, episode))
        item.setData(Qt.ItemDataRole.UserRole, index)
        self.episode_list.addItem(item)
        size = self._episode_size(episode)
        if self.size_filter.findText(size) < 0:
            self.size_filter.addItem(size)
        self._filter_library()
        return episode

    def set_episode(self, episode: ReplayEpisode):
        self.pause()
        self.episode = episode
        self._selected = None
        self._decision_cache = None
        self._inspector_comparison = False
        for view in self.views:
            view.selected_key = None
            view.scene().clearSelection()
        self.position = float(episode.frames[0].get("step", 0)) if episode.frames else 0
        self._reindex()
        for view in (self.xy_view, self.height_view):
            view.set_world_bounds(episode.frames, episode.metadata)
        self._update_metadata()
        self.set_branch_available(not episode.sparse and any(d.get("snapshot") for d in episode.decisions))
        self._update_divergence()
        self.seek(int(self.position))

    replace_episode = set_episode

    def append_episode(self, episode: ReplayEpisode):
        """Replace a live snapshot without resetting camera, selection or time."""
        if self.episode is None:
            self.set_episode(episode)
        else:
            old_episode = self.episode
            self.episode = episode
            for index, (name, entry) in enumerate(self._episodes):
                if entry is old_episode:
                    self._episodes[index] = (name, episode)
                    self.episode_list.item(index).setText(self._episode_caption(name, episode))
            for view in self.views:
                view._trail_cache.clear()
                view._point_cache.clear()
            self._reindex()
            self._update_metadata()
            self.set_branch_available(not episode.sparse and any(d.get("snapshot") for d in episode.decisions))
        if self.follow_live:
            self.seek(episode.max_step)

    def append_frame(self, frame: dict, decision: dict | None = None):
        if self.episode is None:
            self.set_episode(ReplayEpisode(metadata={"dt": 1.0}, frames=[], decisions=[]))
        step = int(frame["step"])
        if self.episode.frames and step < int(self.episode.frames[-1]["step"]):
            raise ValueError("Live frames must arrive in physical-step order")
        if self.episode.frames and step == int(self.episode.frames[-1]["step"]):
            self.episode.frames[-1] = frame
        else:
            self.episode.frames.append(frame)
        if decision is not None:
            self.episode.decisions.append(decision)
        self._reindex()
        if self.follow_live:
            self.seek(step)

    def set_live_mode(self, enabled: bool, *, branching: bool = True):
        self.live_mode = enabled
        self.follow_check.setVisible(enabled)
        self.branch_button.setEnabled(enabled and branching)

    def set_branch_available(self, enabled: bool):
        """Hosts enable this only when the source session can be reconstructed."""
        self.branch_button.setEnabled(enabled)
        self.branch_button.setToolTip("从当前步之前的最近已记录决策边界创建独立分支；原始回合保持不变")

    def _set_follow(self, enabled):
        self.follow_live = enabled
        if enabled and self.episode:
            self.seek(self.episode.max_step)

    def _speed_changed(self, value):
        if self.live_mode:
            self.control_requested.emit("speed:" + value.replace("×", ""))

    def set_comparison(self, episode: ReplayEpisode | None):
        self.comparison = episode
        self.compare_view.setVisible(episode is not None)
        self.comparison_label.setVisible(episode is not None)
        if episode is not None:
            self.compare_view.set_world_bounds(episode.frames, episode.metadata)
            # Use common bounds for a fair visual comparison.
            combined = (self.episode.frames if self.episode else []) + episode.frames
            self.xy_view.set_world_bounds(combined)
            self.compare_view.set_world_bounds(combined)
            self.canvas_splitter.setSizes([500, 500])
        self._update_divergence()
        self.seek(int(self.position))

    def _reindex(self):
        if not self.episode:
            return
        self._steps = [int(f["step"]) for f in self.episode.frames]
        self._decision_steps = sorted({int(d["step"]) for d in self.episode.decisions})
        self.timeline.blockSignals(True)
        self.timeline.setRange(self._steps[0] if self._steps else 0, self.episode.max_step)
        self.timeline.blockSignals(False)
        self.timeline.markers = [(s, "decision") for s in self._decision_steps] + [(int(f["step"]), "event") for f in self.episode.frames if f.get("events")]
        self.timeline.update()
        self.event_list.clear()
        for frame in self.episode.frames:
            for event in frame.get("events", []):
                kind = event.get("type", event.get("kind", "event"))
                source, target = event.get("source_id", "—"), event.get("target_id", "—")
                amount = event.get("damage", event.get("amount"))
                suffix = f"  伤害 {float(amount):.3g}" if isinstance(amount, (int, float)) and amount else ""
                source = "—" if source is None else source
                target = "—" if target is None else target
                item = QListWidgetItem(f"{frame['step']:>5}  {EVENT_NAMES.get(kind, kind)}  {source} → {target}{suffix}")
                item.setData(Qt.ItemDataRole.UserRole, int(frame["step"]))
                item.setToolTip(json.dumps(event, ensure_ascii=False, indent=2, default=str))
                self.event_list.addItem(item)

    def seek(self, step: int):
        if not self.episode or not self.episode.frames:
            return
        self.position = float(max(self._steps[0], min(int(step), self.episode.max_step)))
        self._render_position(False)

    def _render_position(self, smooth=False):
        if not self.episode or not self.episode.frames:
            return
        index = max(1, bisect.bisect_right(self._steps, int(self.position)))
        real = self.episode.frames[index-1]  # The views only read this recorded state.
        frame = real
        if smooth and self.interpolate_check.isChecked() and not self.episode.sparse:
            from .playback import display_frame
            frame = display_frame(self.episode, self.position, interpolate=True)
        self._frame = real
        history = self.episode.frames[max(0, index-60):index]
        self.xy_view.set_frame(frame, history)
        self.height_view.set_frame(frame, history)
        if self.comparison and self.comparison.frames:
            # Compare physical times even when recordings use different dt.
            sim_time = float(real.get("sim_time", int(real["step"]) * self._dt(self.episode)))
            times = [float(f.get("sim_time", f["step"] * self._dt(self.comparison))) for f in self.comparison.frames]
            compare_index = max(0, bisect.bisect_right(times, sim_time + 1e-10)-1)
            compare_step = int(self.comparison.frames[compare_index]["step"])
            comparison_frame = self.comparison.frames[compare_index]
            compare_index = bisect.bisect_right([f["step"] for f in self.comparison.frames], compare_step)
            self.compare_view.set_frame(comparison_frame, self.comparison.frames[max(0, compare_index-60):compare_index])
        self.timeline.blockSignals(True)
        self.timeline.setValue(int(self.position))
        self.timeline.blockSignals(False)
        actual_step = int(real["step"])
        sim_time = float(real.get("sim_time", actual_step * self._dt(self.episode)))
        state = "实时" if self.live_mode else ("播放中" if self.playing else "暂停")
        self.step_label.setText(f"{state}  ·  STEP {actual_step} / {self.episode.max_step}  ·  {sim_time:.2f} s")
        self._update_inspector()
        self._update_decision(actual_step)
        sparse_note = f"稀疏记录：请求步 {int(self.position)}，显示真实采样步 {actual_step}；缺失物理帧不插值。" if self.episode.sparse else "完整物理记录 · 所有位姿单位 m，速度 m/s · 显示操作不推进仿真"
        if not self.xy_view.entities:
            sparse_note += "  此帧未记录可绘制实体位姿。"
        self.statusBar().showMessage(sparse_note)
        self.frame_changed.emit(actual_step)

    @staticmethod
    def _dt(episode):
        for earlier, later in zip(episode.frames, episode.frames[1:]):
            if earlier.get("sim_time") is not None and later.get("sim_time") is not None:
                delta = later["step"] - earlier["step"]
                elapsed = later["sim_time"] - earlier["sim_time"]
                if delta > 0 and elapsed > 0:
                    return float(elapsed / delta)
        value = episode.metadata.get("dt", 1.0)
        return float(value) if isinstance(value, (float, int)) and value > 0 else 1.0

    def toggle_play(self):
        if self.playing:
            self.pause()
            return
        if not self.episode or not self.episode.frames:
            if self.live_mode:
                self.control_requested.emit("play")
            return
        if not self.live_mode and int(self.position) >= self.episode.max_step:
            self.seek(self._steps[0])
        self.playing = True
        self.play_button.setText("暂停")
        self._last_tick = time.perf_counter()
        if self.live_mode:
            self.control_requested.emit("play")

    def pause(self):
        was_playing = self.playing
        self.playing = False
        self.play_button.setText("播放")
        if self.live_mode and was_playing:
            self.control_requested.emit("pause")
        if self.episode and self.episode.frames:
            self.position = float(self.episode.frame_at(int(self.position))["step"])
            self._render_position(False)

    def step_forward(self):
        self.pause()
        if self.live_mode and self.follow_live:
            self.control_requested.emit("step")
        elif self.episode:
            index = bisect.bisect_right(self._steps, int(self.position))
            if index < len(self._steps):
                self.seek(self._steps[index])

    def next_decision(self):
        self.pause()
        if self.live_mode and self.follow_live:
            self.control_requested.emit("next_decision")
            return
        following = next((s for s in self._decision_steps if s > int(self.position)), None)
        if following is not None:
            self.seek(following)

    def next_event(self):
        self.pause()
        if self.live_mode and self.follow_live:
            self.control_requested.emit("next_event")
            return
        if self.episode:
            kinds = {"death", "collision", "agents_destroyed", "target_destroyed", "targets_breached", "self_destruct", "terminal", "red_win", "blue_win"}
            steps = [int(f["step"]) for f in self.episode.frames
                     if any(e.get("kind", e.get("type")) in kinds for e in f.get("events", []))]
            following = next((s for s in steps if s > int(self.position)), None)
            if following is not None:
                self.seek(following)

    def _tick(self):
        now = time.perf_counter()
        elapsed = min(now - self._last_tick, 0.25)
        self._last_tick = now
        if not self.playing or self.live_mode or not self.episode or not self.episode.frames:
            return
        speed = float(self.speed_combo.currentText().replace("×", ""))
        self.position = min(self.episode.max_step, self.position + elapsed * speed / self._dt(self.episode))
        self._render_position(True)
        if self.position >= self.episode.max_step:
            self.pause()

    def select_entity(self, side: str, identifier: str):
        self._selected = (str(side), str(identifier))
        self._inspector_comparison = False
        for view in (self.xy_view, self.height_view):
            view.select_entity(side, identifier)
        self._update_inspector()

    def _update_inspector(self, frame=None, comparison=False):
        self.entity_tree.clear()
        if not self._selected:
            return
        comparison = comparison or self._inspector_comparison
        frame = frame or (self.compare_view.frame if comparison else self._frame)
        entity = next((e for e in frame.get("entities", []) if entity_key(e) == self._selected), None)
        if entity is None:
            self.inspector_title.setText("当前记录帧未提供该实体")
            return
        self.inspector_title.setText(f"{'对比侧 · ' if comparison else ''}{entity['side'].upper()} {entity['id']}  /  {entity.get('role', 'unknown')}")
        position = entity.get("position")
        velocity = entity.get("velocity")
        speed = math.sqrt(sum(float(v)**2 for v in velocity)) if velocity else None
        group = next((g for g in (frame.get("groups", {}).get(entity["side"], []) or []) if str(entity["id"]) in [str(i) for i in g.get("members", [])]), None)
        assignments = (frame.get("assignments", {}).get(entity["side"], {}) or {})
        task = assignments.get(str(entity["id"]), assignments.get(entity["id"], group.get("target_id") if group else None))
        rows = [
            ("实体 ID", entity["id"]), ("角色", entity.get("role")), ("存活", entity.get("alive")),
            ("生命值", entity.get("health")), ("初始生命值", entity.get("max_health")),
            ("位置 / m", position), ("高度 / m", position[2] if position else None),
            ("速度 / m/s", velocity), ("速率 / m/s", round(speed, 3) if speed is not None else None),
            ("编组成员", group.get("members") if group else None), ("目标 / 任务", task),
            ("攻击范围 / m", entity.get("attack_range")),
        ]
        for key, value in rows:
            shown = "未记录" if value is None else json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, float):
                shown = f"{value:.4g}"
            item = QTreeWidgetItem([key, shown])
            item.setToolTip(1, shown)
            self.entity_tree.addTopLevelItem(item)

    def _inspect_comparison(self, side, identifier):
        self._selected = (side, identifier)
        self._inspector_comparison = True
        self._update_inspector(self.compare_view.frame, comparison=True)

    def _update_decision(self, step):
        if not self.episode:
            return
        latest = next((d for d in reversed(self.episode.decisions) if int(d["step"]) <= step), None)
        if latest is not None and self._decision_cache is latest:
            return
        self._decision_cache = latest
        if latest is None:
            text = "该物理步之前未记录上层决策。"
        else:
            plan = latest.get("selected_plan") or {}
            groups = plan.get("groups", []) if isinstance(plan, dict) else []
            reserve = plan.get("reserve", []) if isinstance(plan, dict) else []
            reason = latest.get("event_reason", "未记录")
            reason = {"periodic": "周期到达", "casualty": "伤亡触发", "terminal": "终局"}.get(reason, reason)
            text = f"决策步  {latest['step']}（作用于后续物理步）\n触发  {reason}"
            elapsed = latest.get("decision_seconds")
            if isinstance(elapsed, (int, float)):
                text += f"\n计算耗时  {elapsed * 1000:.2f} ms"
            if latest.get("selected_plan") is not None:
                text += f"\n\n编组  {len(groups)} 组\n预备队  {len(reserve)} 架"
                for index, group in enumerate(groups):
                    members = ", ".join(f"R{i}" for i in group.get("members", []))
                    text += f"\n{index+1}. 目标 {group.get('target_id', group.get('target', '—'))} ← {members}"
                previous = next((d for d in reversed(self.episode.decisions) if int(d["step"]) < int(latest["step"])), None)
                if previous:
                    def assignments(value):
                        value = value if isinstance(value, dict) else {}
                        result = {str(member): group.get("target_id", group.get("target"))
                                  for group in value.get("groups", []) for member in group.get("members", [])}
                        result.update({str(member): "reserve" for member in value.get("reserve", [])})
                        return result
                    before, after = assignments(previous.get("selected_plan")), assignments(plan)
                    changed = sum(before.get(key) != after.get(key) for key in before.keys() | after.keys())
                    text += f"\n\n较上次分配变化  {changed} 架"
            else:
                text += "\n\n当前协议未提供上层编组方案。"
            trace = latest.get("trace") or {}
            if trace:
                text += "\n\n算法解释 / 候选评分\n" + json.dumps(trace, ensure_ascii=False, indent=2, default=str)
            else:
                text += "\n\n算法解释 / 候选评分：未提供。"
        self.decision_summary.setPlainText(text)
        self._show_raw_decision()

    def _show_raw_decision(self):
        if self.decision_tabs.currentIndex() != 1:
            return
        latest = self._decision_cache
        if latest is None:
            self.decision_text.setPlainText("当前步没有决策记录。")
            return
        display = {key: value for key, value in latest.items() if key != "snapshot"}
        if latest.get("snapshot"):
            display["snapshot"] = {"available": True, "note": "完整恢复数据保存在源回合文件中"}
        self.decision_text.setPlainText(json.dumps(display, ensure_ascii=False, indent=2, default=str))

    def _update_metadata(self):
        if not self.episode:
            return
        m = self.episode.metadata
        scenario = m.get("scenario", {}) or {}
        policies = m.get("policies", {}) or {}
        self.metadata_label.setText(
            f"{m.get('episode_id', m.get('name', 'HAD episode'))}   ·   "
            f"{policies.get('red', '未记录')} vs {policies.get('blue', '未记录')}   ·   "
            f"协议 {m.get('protocol_id', '历史 / 未记录')}   ·   核心 {m.get('physics_version', '未记录')}   ·   "
            f"seed {scenario.get('opening_seed', m.get('seed', '未记录'))} / {scenario.get('opponent_seed', '未记录')}"
        )
        self.metadata_label.setToolTip(json.dumps(m, ensure_ascii=False, indent=2, default=str))

    @staticmethod
    def _episode_size(episode):
        entities = episode.frames[0].get("entities", []) if episode.frames else []
        red = sum(e.get("side") == "red" for e in entities)
        blue = sum(e.get("side") == "blue" for e in entities)
        counts = episode.metadata.get("scenario", {}).get("counts", {})
        scenario = episode.metadata.get("scenario", {})
        red, blue = scenario.get("red_count", red), scenario.get("blue_count", blue)
        if isinstance(counts, dict):
            red = counts.get("red", counts.get("red_n", red))
            blue = counts.get("blue", counts.get("blue_n", blue))
        return f"{red}v{blue}"

    @staticmethod
    def _outcome(episode):
        if episode.metadata.get("task_mode", episode.metadata.get("scenario", {}).get("task_mode")) == "damage":
            return 5
        if episode.metadata.get("truncated") or episode.metadata.get("termination_reason") == "sampling_horizon":
            return 4
        result = episode.metadata.get("outcome_red")
        if isinstance(result, str):
            if result.lower() in {"win", "red", "red_win", "victory"}:
                return 1
            if result.lower() in {"loss", "blue", "blue_win", "defeat"}:
                return 2
            return 3
        if isinstance(result, (int, float)) and not isinstance(result, bool):
            return 1 if result > 0 else 2 if result < 0 else 3
        success = episode.metadata.get("success_native")
        return 1 if success == 1 else 2 if success == -1 else 3

    def _episode_caption(self, name, episode):
        m = episode.metadata
        p = m.get("policies", {}) or {}
        category = self._outcome(episode)
        outcome = ["", "红方胜", "蓝方胜", "未决 / 平局", "采样截断", "累计伤害"][category]
        if category == 5:
            total = float(m.get("target_damage", episode.frames[-1].get("target_damage", 0.) if episode.frames else 0.))
            outcome = f"累计伤害 {total:.3f} · 红方回报 {-total:.3f} · 蓝方回报 {total:.3f}"
        reason = m.get("termination_reason", "未记录")
        return f"{name}\n{p.get('red', '?')} × {p.get('blue', '?')} · {self._episode_size(episode)}\n{outcome} · {EVENT_NAMES.get(reason, reason)}"

    def _filter_library(self):
        query = self.filter_text.text().strip().lower()
        for index, (name, episode) in enumerate(self._episodes):
            text = (name + json.dumps(episode.metadata, ensure_ascii=False, default=str)).lower()
            match = (not query or query in text)
            match &= self.outcome_filter.currentIndex() in (0, self._outcome(episode))
            match &= self.size_filter.currentIndex() == 0 or self.size_filter.currentText() == self._episode_size(episode)
            self.episode_list.item(index).setHidden(not match)

    def _library_selected(self, row):
        if row >= 0 and row < len(self._episodes):
            self.set_episode(self._episodes[row][1])

    def _update_divergence(self):
        self._comparison_divergence = None
        self.timeline.markers = [(step, kind) for step, kind in self.timeline.markers if kind != "divergence"]
        if self.comparison is None or self.episode is None:
            return
        left = {float(d["step"]) * self._dt(self.episode): d.get("selected_plan") for d in self.episode.decisions}
        right = {float(d["step"]) * self._dt(self.comparison): d.get("selected_plan") for d in self.comparison.decisions}
        for stamp in sorted(left.keys() | right.keys()):
            if left.get(stamp) != right.get(stamp):
                self._comparison_divergence = int(round(stamp / self._dt(self.episode)))
                self.timeline.markers.append((self._comparison_divergence, "divergence"))
                break
        self.timeline.update()
        same_initial = bool(self.episode.frames and self.comparison.frames and self.episode.frames[0].get("entities") == self.comparison.frames[0].get("entities"))
        label = "同初态" if same_initial else "初态不同：仅作时间对齐查看"
        left_policy = self.episode.metadata.get("policies", {}).get("red", "A")
        right_policy = self.comparison.metadata.get("policies", {}).get("red", "B")
        divergence = f"首次决策分歧：step {self._comparison_divergence}" if self._comparison_divergence is not None else "已记录决策中未发现分歧"
        self.comparison_label.setText(f"左 {left_policy}  /  右 {right_policy}   ·   {label}   ·   {divergence}")

    def set_layer(self, name, enabled):
        for view in self.views:
            view.set_layer(name, enabled)

    def _set_light(self, enabled):
        for view in self.views:
            view.set_light_theme(enabled)

    def _fit_views(self):
        for view in self.views:
            view.fit_world()

    def _event_selected(self, item):
        self.pause()
        self.seek(item.data(Qt.ItemDataRole.UserRole))

    def _open_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "打开 HAD 回合", "", "HAD 记录 (*.json *.json.gz *.jsonl.gz);;所有文件 (*)")
        for path in paths:
            try:
                self.load_episode(path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                QMessageBox.warning(self, "记录载入失败", f"{path}\n{exc}")

    def _open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择回合记录目录")
        if not directory:
            return
        failures = []
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file() and (path.name.endswith(".json") or path.name.endswith(".json.gz") or path.name.endswith(".jsonl.gz")):
                try:
                    self.add_episode(path)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    failures.append(f"{path.name}: {exc}")
        if self._episodes:
            self.episode_list.setCurrentRow(len(self._episodes)-1)
        self.statusBar().showMessage(f"回合库共 {len(self._episodes)} 项；跳过 {len(failures)} 个非回合或无效文件")
        if failures:
            self.episode_list.setToolTip("\n".join(failures[:20]))

    def _compare_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择对比回合", "", "HAD 记录 (*.json *.json.gz *.jsonl.gz)")
        if path:
            try:
                self.set_comparison(load_episode(path))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                QMessageBox.warning(self, "对比载入失败", str(exc))

    def _export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出当前战场", "had-frame.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if path:
            try:
                self.export_scene(path, light=True)
                self.statusBar().showMessage(f"已导出 {path}")
            except (OSError, ValueError, RuntimeError) as exc:
                QMessageBox.warning(self, "导出失败", str(exc))

    def export_scene(self, path, light: bool = False):
        """Export the actual current projection with protocol/step attribution."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in {".png", ".svg", ".pdf"}:
            raise ValueError("Scene export supports .png, .svg or .pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        width = 1600
        main_width = width/2 if self.comparison else width
        main_height = round(main_width * self.xy_view.viewport().height() / max(1, self.xy_view.viewport().width()))
        profile_height = round(width * self.height_view.viewport().height() / max(1, self.height_view.viewport().width()))
        height = 120 + main_height + profile_height
        if suffix == ".png":
            device = QImage(width, height, QImage.Format.Format_ARGB32)
        elif suffix == ".svg":
            device = QSvgGenerator()
            device.setFileName(str(path))
            device.setSize(QSize(width, height))
            device.setViewBox(QRectF(0, 0, width, height))
            device.setTitle("HAD recorded battlefield")
        else:
            device = QPdfWriter(str(path))
            device.setResolution(120)
            device.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
            device.setPageOrientation(QPageLayout.Orientation.Landscape)
        painter = QPainter(device)
        if not painter.isActive():
            raise OSError(f"Cannot create export {path}")
        old_theme = [view.light for view in self.views]
        try:
            for view in self.views:
                view.set_light_theme(light)
            if suffix == ".pdf":
                painter.fillRect(QRectF(0, 0, device.width(), device.height()), QColor("#ffffff" if light else "#111c2d"))
                factor = min(device.width()/width, device.height()/height)
                painter.translate((device.width()-width*factor)/2, (device.height()-height*factor)/2)
                painter.scale(factor, factor)
            painter.fillRect(QRectF(0, 0, width, height), QColor("#ffffff" if light else "#111c2d"))
            painter.setPen(QColor("#233850" if light else "#dce6f3"))
            painter.setFont(QFont("Microsoft YaHei UI", 15))
            painter.drawText(QPointF(24, 32), "HAD / recorded battlefield")
            painter.setFont(QFont("Microsoft YaHei UI", 9))
            painter.drawText(QRectF(24, 45, width-48, 52), Qt.TextFlag.TextWordWrap, self.metadata_label.text() + "\n" + self.step_label.text())
            self.xy_view.render(painter, QRectF(0, 105, main_width, main_height), self.xy_view.viewport().rect(), Qt.AspectRatioMode.KeepAspectRatio)
            if self.comparison:
                self.compare_view.render(painter, QRectF(main_width, 105, main_width, main_height), self.compare_view.viewport().rect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.height_view.render(painter, QRectF(0, 110+main_height, width, profile_height), self.height_view.viewport().rect(), Qt.AspectRatioMode.KeepAspectRatio)
        finally:
            painter.end()
            for view, old in zip(self.views, old_theme):
                view.set_light_theme(old)
        if suffix == ".png" and not device.save(str(path)):
            raise OSError(f"Cannot write {path}")
        return path

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


def run_gui(episode: ReplayEpisode | None = None):
    app = QApplication.instance() or QApplication([])
    window = WorkbenchWindow(episode)
    window.show()
    return app.exec()
