from __future__ import annotations

import csv
import io
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QDate, QObject, QRunnable, QSettings, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QRadioButton,
    QStackedWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pyCheck.cross_day_plots import generate_cross_day_plots
from pyCheck.day_presentation_plots import generate_day_presentation_plots
from pyCheck.joystick_validation import (
    build_replay_session,
    build_validation_report,
    get_trial_segment,
    load_joystick_dataset,
    parse_trial_tokens,
    plot_trial_timeseries,
    plot_trial_trajectory,
    render_trial_replay_frames,
    render_trial_replay_video,
)
from py_proc.run_day_pipeline import run_day_pipeline


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _numpy_bgr_to_pixmap(frame: Any) -> QPixmap:
    import numpy as np
    from PyQt6.QtGui import QImage
    rgb = frame[:, :, ::-1].copy()  # BGR → RGB, ensure contiguous
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class WorkerSignals(QObject):
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)


class Worker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                result = self.fn(*self.args, **self.kwargs)
        except Exception:
            self.signals.error.emit(buffer.getvalue() + "\n" + traceback.format_exc())
            return
        self.signals.finished.emit(result, buffer.getvalue())


class ImagePreview(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.label = QLabel("No image selected")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(QSize(400, 300))
        self.setWidget(self.label)
        self._pixmap: Optional[QPixmap] = None

    def set_image(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.label.setText(f"Could not load image:\n{image_path}")
            self._pixmap = None
            return
        self._pixmap = pixmap
        self._refresh_scaled_pixmap()

    def clear_image(self, text: str = "No image selected") -> None:
        self._pixmap = None
        self.label.setPixmap(QPixmap())
        self.label.setText(text)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def _refresh_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        viewport_size = self.viewport().size()
        scaled = self._pixmap.scaled(
            viewport_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled)


class VideoDisplay(QLabel):
    def __init__(self) -> None:
        super().__init__("No replay rendered yet")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QSize(300, 300))
        self._pixmap: Optional[QPixmap] = None

    def set_frame(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._refresh_scaled()

    def clear_frame(self, text: str = "No replay rendered yet") -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._refresh_scaled()

    def _refresh_scaled(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class ProcGuiWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("proc_gui")
        self.resize(1500, 900)
        self.thread_pool = QThreadPool.globalInstance()
        self.current_image: Optional[Path] = None
        self.output_dir_path: Optional[Path] = None
        self.last_auto_out_dir: Optional[Path] = None
        self.last_auto_cross_day_out_dir: Optional[Path] = None
        self.auto_plot_timer = QTimer(self)
        self.auto_plot_timer.setInterval(350)
        self.auto_plot_timer.setSingleShot(True)
        self.auto_plot_timer.timeout.connect(self.generate_selected_trial_plot)
        self._replay_session: Optional[Any] = None
        self._replay_frame_idx: int = 0
        self._replay_fps: int = 30
        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._replay_tick)
        self._monkey_dir: Optional[Path] = None
        _s = QSettings("PesaranLab", "ProcGui")
        _saved = _s.value("ProcGui/monkey_dir", "")
        if _saved:
            _candidate = Path(_saved).resolve()
            if _candidate.is_dir():
                self._monkey_dir = _candidate
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        controls_layout.addWidget(left_splitter)

        inputs_container = QWidget()
        inputs_layout = QVBoxLayout(inputs_container)

        io_group = QGroupBox("Inputs")
        io_form = QFormLayout(io_group)

        self.day_dir_edit = QLineEdit()
        self.day_dir_edit.editingFinished.connect(self._sync_day_dir_fields)
        browse_day_btn = QPushButton("Browse")
        browse_day_btn.clicked.connect(self._browse_day_dir)
        day_row = QHBoxLayout()
        day_row.addWidget(self.day_dir_edit)
        day_row.addWidget(browse_day_btn)
        io_form.addRow("Day Dir", self._wrap_layout(day_row))

        self.out_dir_edit = QLineEdit()
        browse_out_btn = QPushButton("Browse")
        browse_out_btn.clicked.connect(self._browse_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir_edit)
        out_row.addWidget(browse_out_btn)
        io_form.addRow("Out Dir", self._wrap_layout(out_row))

        inputs_layout.addWidget(io_group)
        inputs_layout.addStretch(1)

        notes_group = QGroupBox("Day Notes")
        notes_layout = QVBoxLayout(notes_group)
        self.notes_path_label = QLabel("No day notes detected")
        self.notes_path_label.setWordWrap(True)
        notes_layout.addWidget(self.notes_path_label)
        self.notes_view = QTextBrowser()
        self.notes_view.setOpenExternalLinks(True)
        self.notes_view.setPlaceholderText("Selecting a day directory will load the day notes markdown file here.")
        notes_layout.addWidget(self.notes_view)

        actions_container = QWidget()
        actions_layout = QVBoxLayout(actions_container)

        tabs = QTabWidget()
        actions_layout.addWidget(tabs)

        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)

        processing_group = QGroupBox("run_day_pipeline")
        processing_form = QFormLayout(processing_group)

        self.pipeline_rec_edit = QLineEdit()
        self.pipeline_rec_edit.setPlaceholderText("blank = all recs, or 005")
        processing_form.addRow("Pipeline Rec", self.pipeline_rec_edit)

        self.skip_video_checkbox = QCheckBox("Skip video extraction")
        processing_form.addRow("", self.skip_video_checkbox)

        self.skip_existing_video_checkbox = QCheckBox("Skip existing video outputs")
        processing_form.addRow("", self.skip_existing_video_checkbox)

        steps_container = QWidget()
        steps_layout = QVBoxLayout(steps_container)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(2)
        self.step_extract_checkbox = QCheckBox("Step 1: Extract")
        self.step_events_nodisplay_checkbox = QCheckBox("Step 2: Events (no-display)")
        self.step_detect_display_checkbox = QCheckBox("Step 3: Detect display states")
        self.step_full_proc_checkbox = QCheckBox("Step 4: Full processing")
        self.step_save_trials_checkbox = QCheckBox("Step 5: Save trials")
        for cb in (
            self.step_extract_checkbox,
            self.step_events_nodisplay_checkbox,
            self.step_detect_display_checkbox,
            self.step_full_proc_checkbox,
            self.step_save_trials_checkbox,
        ):
            cb.setChecked(True)
            steps_layout.addWidget(cb)
        processing_form.addRow("Steps", steps_container)

        processing_layout.addWidget(processing_group)

        processing_button_group = QGroupBox("Processing Actions")
        processing_button_layout = QVBoxLayout(processing_button_group)
        run_pipeline_btn = QPushButton("Run Day Pipeline")
        run_pipeline_btn.clicked.connect(self.run_processing_pipeline)
        processing_button_layout.addWidget(run_pipeline_btn)
        processing_layout.addWidget(processing_button_group)
        processing_layout.addStretch(1)

        tabs.addTab(processing_tab, "Processing")

        inspection_tab = QWidget()
        inspection_layout = QVBoxLayout(inspection_tab)

        summary_group = QGroupBox("Day Summary")
        summary_form = QFormLayout(summary_group)

        self.summary_task_types_edit = QLineEdit()
        self.summary_task_types_edit.setPlaceholderText("joystick_intro, other_task")
        summary_form.addRow("Task Types", self.summary_task_types_edit)

        self.summary_exclude_recs_edit = QLineEdit()
        self.summary_exclude_recs_edit.setPlaceholderText("3,5")
        summary_form.addRow("Exclude Recs", self.summary_exclude_recs_edit)

        summary_btn = QPushButton("Generate Day Summary")
        summary_btn.clicked.connect(self.generate_day_summary)
        summary_form.addRow("", summary_btn)
        self._set_help_text(
            summary_btn,
            (
                "Generate the full day-level summary figure set.\n\n"
                "Uses: Task Types, Exclude Recs, Day Dir, and Out Dir.\n\n"
                "Outputs:\n"
                f"- <day>_overview_by_rec.png: trial count and success rate for each recording.\n"
                f"- <day>_performance_over_time.png: rolling success and trial duration across the session.\n"
                f"- <day>_display_alignment.png: disStartOn latency distribution and per-recording alignment.\n"
                f"- <day>_target_performance.png: spatial target success map and target usage density.\n"
                f"- <day>_success_failure_timing.png: timing distributions for success vs failure.\n"
                f"- <day>_summary_metrics.json: summary values behind the plots.\n\n"
                "Use this when you want a high-level view of session quality, performance drift, target-specific "
                "behavior, and display timing."
            ),
        )
        inspection_layout.addWidget(summary_group)

        validation_group = QGroupBox("Validation Report")
        validation_form = QFormLayout(validation_group)

        self.validation_rec_combo = QComboBox()
        self.validation_rec_combo.setEditable(True)
        validation_form.addRow("Rec", self.validation_rec_combo)

        self.validation_sample_trials_edit = QLineEdit()
        self.validation_sample_trials_edit.setPlaceholderText("1 2 4 or 90-130")
        validation_form.addRow("Sample Trials", self.validation_sample_trials_edit)

        validation_btn = QPushButton("Generate Validation Report")
        validation_btn.clicked.connect(self.generate_validation_report)
        validation_form.addRow("", validation_btn)
        self._set_help_text(
            validation_btn,
            (
                "Generate joystick validation figures for the selected recording.\n\n"
                "Uses: Rec, Sample Trials, Day Dir, and Out Dir.\n\n"
                "Outputs:\n"
                "- alignment_summary.png: distributions of event timing error and cursor reconstruction error.\n"
                "- cursor_shift_sweep.png: how a constant timestamp shift changes cursor error; useful for "
                "spotting systematic timing offsets.\n"
                "- trial_<n>_trajectory.png and trial_<n>_timeseries.png for sample trials.\n"
                "- validation_summary.json: numeric summary including the best constant shift estimate.\n\n"
                "Use this to check whether behavioral events, joystick timestamps, and reconstructed cursor "
                "trajectories are internally consistent."
            ),
        )
        inspection_layout.addWidget(validation_group)

        trial_group = QGroupBox("Trial Plot")
        trial_form = QFormLayout(trial_group)

        self.trial_rec_combo = QComboBox()
        self.trial_rec_combo.setEditable(True)
        self.trial_rec_combo.currentTextChanged.connect(self._schedule_auto_trial_plot)
        trial_form.addRow("Rec", self.trial_rec_combo)

        self.trial_spin = QSpinBox()
        self.trial_spin.setRange(1, 10000)
        self.trial_spin.valueChanged.connect(self._schedule_auto_trial_plot)
        trial_form.addRow("Trial", self.trial_spin)

        self.plot_kind_combo = QComboBox()
        self.plot_kind_combo.addItems(["timeseries", "trajectory"])
        self.plot_kind_combo.currentTextChanged.connect(self._schedule_auto_trial_plot)
        trial_form.addRow("Plot", self.plot_kind_combo)
        self._set_help_text(
            self.plot_kind_combo,
            (
                "Choose the trial figure type.\n\n"
                "timeseries: joystick_x / joystick_y and reconstructed cursor_x / cursor_y over time, "
                "with target position and event markers. Use this to inspect timing, movement onset, "
                "target entry, hold, and reward timing.\n\n"
                "trajectory: reconstructed cursor path in task space with the target circle and labeled "
                "trial events. Use this to inspect spatial path shape, target approach, and endpoint."
            ),
        )

        self.auto_refresh_checkbox = QCheckBox("Auto-refresh trial plot")
        self.auto_refresh_checkbox.setChecked(True)
        trial_form.addRow("", self.auto_refresh_checkbox)
        self._set_help_text(
            self.auto_refresh_checkbox,
            "Automatically regenerate the selected-trial figure when the recording, trial, or plot type changes.",
        )

        trial_btn = QPushButton("Generate Selected Trial Plot")
        trial_btn.clicked.connect(self.generate_selected_trial_plot)
        trial_form.addRow("", trial_btn)
        self._set_help_text(
            trial_btn,
            (
                "Generate one figure for the selected trial and plot type.\n\n"
                "Uses: Rec, Trial, Plot, Day Dir, and Out Dir.\n\n"
                "If Plot = timeseries, the output is trial_<n>_timeseries.png showing joystick and reconstructed "
                "cursor signals over time with event markers.\n"
                "If Plot = trajectory, the output is trial_<n>_trajectory.png showing the cursor path in task "
                "space relative to the target.\n\n"
                "Use this for trial-by-trial inspection after you identify an interesting recording or trial."
            ),
        )

        inspection_layout.addWidget(trial_group)

        refresh_btn = QPushButton("Refresh Output Browser")
        refresh_btn.clicked.connect(self.refresh_output_browser)
        self._set_help_text(
            refresh_btn,
            "Reload the image list from the current output directory and update the preview pane.",
        )
        inspection_layout.addWidget(refresh_btn)
        inspection_layout.addStretch(1)

        tabs.addTab(inspection_tab, "Inspection")

        replay_tab = QWidget()
        replay_layout = QVBoxLayout(replay_tab)

        replay_controls_group = QGroupBox("Replay Controls")
        replay_controls_form = QFormLayout(replay_controls_group)

        self.replay_rec_combo = QComboBox()
        self.replay_rec_combo.setEditable(True)
        replay_controls_form.addRow("Rec", self.replay_rec_combo)

        # Mode selection
        mode_row = QHBoxLayout()
        self.replay_mode_trials = QRadioButton("Trials")
        self.replay_mode_trials.setChecked(True)
        self.replay_mode_time = QRadioButton("Time Window")
        self._replay_mode_group = QButtonGroup(self)
        self._replay_mode_group.addButton(self.replay_mode_trials, 0)
        self._replay_mode_group.addButton(self.replay_mode_time, 1)
        mode_row.addWidget(self.replay_mode_trials)
        mode_row.addWidget(self.replay_mode_time)
        mode_row.addStretch(1)
        replay_controls_form.addRow("Mode", self._wrap_layout(mode_row))

        # Stacked inputs: page 0 = trials, page 1 = time window
        self._replay_input_stack = QStackedWidget()

        trials_page = QWidget()
        trials_page_layout = QHBoxLayout(trials_page)
        trials_page_layout.setContentsMargins(0, 0, 0, 0)
        self.replay_trials_edit = QLineEdit()
        self.replay_trials_edit.setPlaceholderText("1  or  1-5  or  1 3 7")
        trials_page_layout.addWidget(self.replay_trials_edit)
        self._replay_input_stack.addWidget(trials_page)

        time_page = QWidget()
        time_page_layout = QHBoxLayout(time_page)
        time_page_layout.setContentsMargins(0, 0, 0, 0)
        self.replay_start_spin = QDoubleSpinBox()
        self.replay_start_spin.setRange(0.0, 99999.0)
        self.replay_start_spin.setDecimals(2)
        self.replay_start_spin.setSuffix(" s")
        self.replay_start_spin.setToolTip("Seconds from the start of the recording's first joystick sample")
        self.replay_duration_spin = QDoubleSpinBox()
        self.replay_duration_spin.setRange(0.1, 600.0)
        self.replay_duration_spin.setDecimals(1)
        self.replay_duration_spin.setValue(30.0)
        self.replay_duration_spin.setSuffix(" s")
        time_page_layout.addWidget(QLabel("Start"))
        time_page_layout.addWidget(self.replay_start_spin)
        time_page_layout.addSpacing(12)
        time_page_layout.addWidget(QLabel("Duration"))
        time_page_layout.addWidget(self.replay_duration_spin)
        time_page_layout.addStretch(1)
        self._replay_input_stack.addWidget(time_page)

        self._replay_mode_group.idToggled.connect(
            lambda bid, checked: self._replay_input_stack.setCurrentIndex(bid) if checked else None
        )
        replay_controls_form.addRow("", self._replay_input_stack)

        fps_speed_row = QHBoxLayout()
        self.replay_fps_spin = QSpinBox()
        self.replay_fps_spin.setRange(1, 60)
        self.replay_fps_spin.setValue(20)
        fps_speed_row.addWidget(QLabel("FPS"))
        fps_speed_row.addWidget(self.replay_fps_spin)
        fps_speed_row.addSpacing(16)
        self.replay_speed_spin = QDoubleSpinBox()
        self.replay_speed_spin.setRange(0.1, 10.0)
        self.replay_speed_spin.setSingleStep(0.5)
        self.replay_speed_spin.setValue(1.0)
        fps_speed_row.addWidget(QLabel("Speed"))
        fps_speed_row.addWidget(self.replay_speed_spin)
        fps_speed_row.addStretch(1)
        replay_controls_form.addRow("", self._wrap_layout(fps_speed_row))

        render_replay_btn = QPushButton("Render")
        render_replay_btn.clicked.connect(self.render_trial_replay)
        self._set_help_text(
            render_replay_btn,
            (
                "Render the selected trials or time window into memory.\n\n"
                "Trials mode: enter a range like 5-12 or individual numbers like 1 3 7.\n"
                "Time Window mode: enter start (seconds from rec start) and duration.\n\n"
                "Camera panels (Cam Top, Cam Left) stream live from the AVI files during playback.\n"
                "Joystick overlay frames are pre-rendered; playback starts automatically."
            ),
        )
        replay_controls_form.addRow("", render_replay_btn)

        self.replay_frame_info_label = QLabel("No frames rendered")
        replay_controls_form.addRow("", self.replay_frame_info_label)

        replay_layout.addWidget(replay_controls_group)

        # 3-panel display: joystick overlay | Cam Top / Cam Left
        display_splitter = QSplitter(Qt.Orientation.Horizontal)

        mpl_panel = QWidget()
        mpl_layout = QVBoxLayout(mpl_panel)
        mpl_layout.setContentsMargins(0, 0, 0, 0)
        mpl_layout.addWidget(QLabel("Joystick Overlay"))
        self.replay_display = VideoDisplay()
        mpl_layout.addWidget(self.replay_display, 1)
        display_splitter.addWidget(mpl_panel)

        cam_splitter = QSplitter(Qt.Orientation.Vertical)

        cam_top_panel = QWidget()
        cam_top_layout = QVBoxLayout(cam_top_panel)
        cam_top_layout.setContentsMargins(0, 0, 0, 0)
        cam_top_layout.addWidget(QLabel("Cam Top"))
        self.replay_cam_top = VideoDisplay()
        self.replay_cam_top.clear_frame("No camera data")
        cam_top_layout.addWidget(self.replay_cam_top, 1)
        cam_splitter.addWidget(cam_top_panel)

        cam_left_panel = QWidget()
        cam_left_layout = QVBoxLayout(cam_left_panel)
        cam_left_layout.setContentsMargins(0, 0, 0, 0)
        cam_left_layout.addWidget(QLabel("Cam Left"))
        self.replay_cam_left = VideoDisplay()
        self.replay_cam_left.clear_frame("No camera data")
        cam_left_layout.addWidget(self.replay_cam_left, 1)
        cam_splitter.addWidget(cam_left_panel)

        display_splitter.addWidget(cam_splitter)
        display_splitter.setStretchFactor(0, 1)
        display_splitter.setStretchFactor(1, 2)
        replay_layout.addWidget(display_splitter, 1)

        playback_row = QHBoxLayout()
        self.replay_play_btn = QPushButton("Play")
        self.replay_play_btn.setEnabled(False)
        self.replay_play_btn.clicked.connect(self._replay_play)
        self.replay_stop_btn = QPushButton("Stop")
        self.replay_stop_btn.setEnabled(False)
        self.replay_stop_btn.clicked.connect(self._replay_stop)
        self.replay_replay_btn = QPushButton("Replay")
        self.replay_replay_btn.setEnabled(False)
        self.replay_replay_btn.clicked.connect(self._replay_replay)
        self.replay_download_btn = QPushButton("Download MP4")
        self.replay_download_btn.setEnabled(False)
        self.replay_download_btn.clicked.connect(self._replay_download)
        playback_row.addWidget(self.replay_play_btn)
        playback_row.addWidget(self.replay_stop_btn)
        playback_row.addWidget(self.replay_replay_btn)
        playback_row.addStretch(1)
        playback_row.addWidget(self.replay_download_btn)
        replay_layout.addLayout(playback_row)

        tabs.addTab(replay_tab, "Replay")

        cross_day_tab = QWidget()
        cross_day_layout = QVBoxLayout(cross_day_tab)

        cross_day_group = QGroupBox("Cross-Day Summary")
        cross_day_form = QFormLayout(cross_day_group)

        self.cross_day_recs_edit = QLineEdit()
        self.cross_day_recs_edit.setPlaceholderText("1-4,6 or blank = all recs in current day")
        cross_day_form.addRow("Current Day Recs", self.cross_day_recs_edit)

        self.cross_day_available_recs_label = QLabel("Available recs: none")
        self.cross_day_available_recs_label.setWordWrap(True)
        cross_day_form.addRow("", self.cross_day_available_recs_label)

        add_day_btn = QPushButton("Add / Update Current Day")
        add_day_btn.clicked.connect(self.add_cross_day_selection)
        cross_day_form.addRow("", add_day_btn)
        self._set_help_text(
            add_day_btn,
            (
                "Add the currently selected day directory to the cross-day comparison list.\n\n"
                "Uses the Current Day Recs field. Leave it blank to include all recordings found for that day.\n"
                "Adding the same day again updates its recording selection."
            ),
        )

        self.cross_day_selection_list = QListWidget()
        self.cross_day_selection_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.cross_day_selection_list.setMinimumHeight(240)
        cross_day_form.addRow("Selected Days", self.cross_day_selection_list)

        cross_day_button_row = QHBoxLayout()
        remove_day_btn = QPushButton("Remove Selected")
        remove_day_btn.clicked.connect(self.remove_cross_day_selection)
        clear_days_btn = QPushButton("Clear All")
        clear_days_btn.clicked.connect(self.clear_cross_day_selections)
        cross_day_button_row.addWidget(remove_day_btn)
        cross_day_button_row.addWidget(clear_days_btn)
        cross_day_form.addRow("", self._wrap_layout(cross_day_button_row))

        self.cross_day_task_types_edit = QLineEdit()
        self.cross_day_task_types_edit.setPlaceholderText("joystick_intro, other_task")
        cross_day_form.addRow("Task Types", self.cross_day_task_types_edit)

        self.cross_day_out_dir_edit = QLineEdit()
        browse_cross_day_out_btn = QPushButton("Browse")
        browse_cross_day_out_btn.clicked.connect(self._browse_cross_day_out_dir)
        cross_day_out_row = QHBoxLayout()
        cross_day_out_row.addWidget(self.cross_day_out_dir_edit)
        cross_day_out_row.addWidget(browse_cross_day_out_btn)
        cross_day_form.addRow("Cross-Day Out", self._wrap_layout(cross_day_out_row))

        cross_day_btn = QPushButton("Generate Cross-Day Summary")
        cross_day_btn.clicked.connect(self.generate_cross_day_summary)
        cross_day_form.addRow("", cross_day_btn)
        self._set_help_text(
            cross_day_btn,
            (
                "Generate cross-day summary figures and metrics for the selected day/recording set.\n\n"
                "Outputs:\n"
                "- cross_day_metrics.png: duration, post-entry stability, path efficiency, and successful trials.\n"
                "- cross_day_summary_metrics.json: selected-run summary.\n"
                "- cross_day_summary_metrics.csv: cumulative day-by-day metrics that are preserved and updated over time.\n\n"
                "Use this when you want to compare learning across days while keeping a running CSV as more sessions are collected."
            ),
        )

        cross_day_layout.addWidget(cross_day_group)
        cross_day_layout.addStretch(1)

        tabs.addTab(cross_day_tab, "Cross Day")

        self.status_label = QLabel("Ready")
        actions_layout.addWidget(self.status_label)
        actions_layout.addStretch(1)

        log_group = QGroupBox("Output Log")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)

        repo_group = QGroupBox("Monkey Repo")
        repo_layout = QVBoxLayout(repo_group)

        monkey_dir_row = QHBoxLayout()
        self.monkey_dir_edit = QLineEdit()
        self.monkey_dir_edit.setPlaceholderText("Monkey directory  (e.g. /cdz/pesaranlab/Eevee_Behavior_AlexRig)")
        self.monkey_dir_edit.editingFinished.connect(self._on_monkey_dir_changed)
        browse_monkey_btn = QPushButton("Browse")
        browse_monkey_btn.clicked.connect(self._browse_monkey_dir)
        refresh_repo_btn = QPushButton("Refresh")
        refresh_repo_btn.clicked.connect(self._refresh_day_table)
        monkey_dir_row.addWidget(self.monkey_dir_edit)
        monkey_dir_row.addWidget(browse_monkey_btn)
        monkey_dir_row.addWidget(refresh_repo_btn)
        repo_layout.addLayout(monkey_dir_row)

        self.day_status_table = QTableWidget(0, 5)
        self.day_status_table.setHorizontalHeaderLabels(["Day", "Recs", "Events", "AllTrials", "Summary"])
        self.day_status_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.day_status_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.day_status_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.day_status_table.verticalHeader().setVisible(False)
        hh = self.day_status_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.day_status_table.cellClicked.connect(self._on_day_table_row_clicked)
        repo_layout.addWidget(self.day_status_table)

        add_cross_day_btn = QPushButton("Add Selected to Cross-Day")
        add_cross_day_btn.setToolTip("Add all highlighted rows to the Cross-Day selection list (Ctrl+click to select multiple)")
        add_cross_day_btn.clicked.connect(self._add_table_days_to_cross_day)
        repo_layout.addWidget(add_cross_day_btn)

        if self._monkey_dir is not None:
            self.monkey_dir_edit.setText(str(self._monkey_dir))

        left_splitter.addWidget(repo_group)
        left_splitter.addWidget(inputs_container)
        left_splitter.addWidget(notes_group)
        left_splitter.addWidget(actions_container)
        left_splitter.addWidget(log_group)
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 2)
        left_splitter.setStretchFactor(2, 3)
        left_splitter.setStretchFactor(3, 5)
        left_splitter.setStretchFactor(4, 2)
        left_splitter.setSizes([220, 130, 200, 380, 150])

        splitter.addWidget(controls_panel)

        right_splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(right_splitter)

        self.image_list = QListWidget()
        self.image_list.itemSelectionChanged.connect(self._show_selected_image)
        right_splitter.addWidget(self.image_list)

        self.preview = ImagePreview()
        right_splitter.addWidget(self.preview)

        splitter.setSizes([460, 1040])
        right_splitter.setSizes([260, 780])

        if self._monkey_dir is not None:
            self._refresh_day_table()

    def _wrap_layout(self, layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _set_help_text(self, widget: QWidget, text: str) -> None:
        widget.setToolTip(text)
        widget.setStatusTip(text.replace("\n", " "))
        widget.setWhatsThis(text)
        widget.setToolTipDuration(20000)

    def _browse_monkey_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose monkey repo directory")
        if path:
            self.monkey_dir_edit.setText(path)
            self._on_monkey_dir_changed()

    def _on_monkey_dir_changed(self) -> None:
        text = self.monkey_dir_edit.text().strip()
        if not text:
            self._monkey_dir = None
            return
        candidate = Path(text).resolve()
        if not candidate.is_dir():
            return
        self._monkey_dir = candidate
        self._refresh_day_table()

    def _scan_day_status(self, monkey_dir: Path) -> list[dict]:
        days = sorted(
            [p for p in monkey_dir.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: p.name,
            reverse=True,
        )
        results = []
        for day_path in days:
            day = day_path.name
            rec_dirs = sorted(p.name for p in day_path.iterdir() if p.is_dir() and p.name.isdigit())
            events_complete = [
                rec for rec in rec_dirs
                if (day_path / rec / f"rec{rec}.Events.mat").exists()
            ]
            all_trials = (day_path / "mat" / "AllTrials.mat").exists()
            summary_plots = (monkey_dir / "claude" / "figures" / day / "beh" / f"{day}_overview_by_rec.png").exists()
            has_notes = bool(list(day_path.glob(f"{day}_*.md")))
            results.append({
                "day": day,
                "recs": rec_dirs,
                "events_complete": events_complete,
                "all_trials": all_trials,
                "summary_plots": summary_plots,
                "has_notes": has_notes,
            })
        return results

    def _refresh_day_table(self) -> None:
        monkey_dir = self._monkey_dir
        if monkey_dir is None or not monkey_dir.is_dir():
            self.day_status_table.setRowCount(0)
            return

        day_records = self._scan_day_status(monkey_dir)
        self._write_processing_index(monkey_dir, day_records)

        current_day = ""
        current_day_dir = self._day_dir_path()
        if current_day_dir is not None and current_day_dir.parent == monkey_dir:
            current_day = current_day_dir.name
        if not current_day:
            today = QDate.currentDate().toString("yyMMdd")
            if any(r["day"] == today for r in day_records):
                current_day = today
            elif day_records:
                current_day = day_records[0]["day"]

        self.day_status_table.blockSignals(True)
        self.day_status_table.setRowCount(len(day_records))

        def _cell(text: str) -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            return item

        highlight_row = -1
        for row, rec in enumerate(day_records):
            day = rec["day"]
            recs = rec["recs"]
            n_recs = len(recs)
            recs_text = str(n_recs) if n_recs > 0 else "—"
            events_text = f"{len(rec['events_complete'])}/{n_recs}" if n_recs > 0 else "—"
            all_trials_text = "✓" if rec["all_trials"] else "—"
            summary_text = "✓" if rec["summary_plots"] else "—"
            self.day_status_table.setItem(row, 0, _cell(day))
            self.day_status_table.setItem(row, 1, _cell(recs_text))
            self.day_status_table.setItem(row, 2, _cell(events_text))
            self.day_status_table.setItem(row, 3, _cell(all_trials_text))
            self.day_status_table.setItem(row, 4, _cell(summary_text))
            if day == current_day:
                highlight_row = row

        self.day_status_table.blockSignals(False)

        if highlight_row >= 0:
            self.day_status_table.selectRow(highlight_row)

        if current_day_dir is None and current_day and day_records and highlight_row >= 0:
            self._on_day_table_row_clicked(highlight_row, 0)

    def _on_day_table_row_clicked(self, row: int, col: int) -> None:
        if len(self.day_status_table.selectionModel().selectedRows()) != 1:
            return
        day_item = self.day_status_table.item(row, 0)
        if day_item is None or self._monkey_dir is None:
            return
        day = day_item.text()
        day_path = self._monkey_dir / day
        if not day_path.is_dir():
            return
        self.day_dir_edit.setText(str(day_path))
        self._sync_day_dir_fields()

    def _add_table_days_to_cross_day(self) -> None:
        if self._monkey_dir is None:
            return
        selected_rows = self.day_status_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "proc_gui", "Select one or more days in the table first.")
            return
        for index in selected_rows:
            day_item = self.day_status_table.item(index.row(), 0)
            if day_item is None:
                continue
            day = day_item.text()
            existing_row = None
            for i in range(self.cross_day_selection_list.count()):
                if self.cross_day_selection_list.item(i).text().split(":", 1)[0].strip() == day:
                    existing_row = i
                    break
            if existing_row is None:
                self.cross_day_selection_list.addItem(day)
            # if already present, leave it unchanged (preserves any rec spec the user set)

    def _write_processing_index(self, monkey_dir: Path, day_records: list[dict]) -> None:
        index_path = monkey_dir / "claude" / "processing_index.json"
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "monkey_dir": str(monkey_dir),
                "last_updated": datetime.now().isoformat(timespec="seconds"),
                "days": {
                    rec["day"]: {
                        "recs": rec["recs"],
                        "events_complete": rec["events_complete"],
                        "all_trials": rec["all_trials"],
                        "summary_plots": rec["summary_plots"],
                        "has_notes": rec["has_notes"],
                    }
                    for rec in day_records
                },
            }
            index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self._append_log(f"Warning: could not write processing_index.json: {exc}")

    def _highlight_day_table_row(self, day_name: str) -> None:
        for row in range(self.day_status_table.rowCount()):
            item = self.day_status_table.item(row, 0)
            if item is not None and item.text() == day_name:
                self.day_status_table.blockSignals(True)
                self.day_status_table.selectRow(row)
                self.day_status_table.blockSignals(False)
                return

    def _browse_day_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose day directory")
        if path:
            self.day_dir_edit.setText(path)
            self._sync_day_dir_fields()

    def _browse_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if path:
            self.out_dir_edit.setText(path)
            self.output_dir_path = Path(path).resolve()
            self.refresh_output_browser()

    def _browse_cross_day_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose cross-day output directory")
        if path:
            self.cross_day_out_dir_edit.setText(path)
            self.last_auto_cross_day_out_dir = Path(path).resolve()
            self._load_cross_day_selections_from_csv()

    def _sync_day_dir_fields(self) -> None:
        day_dir = self._day_dir_path()
        if day_dir is None:
            return
        repo_root = day_dir.parent
        expected_out_dir = repo_root / "claude" / "figures" / day_dir.name / "beh"
        current_out_dir = self._out_dir_path()
        should_auto_update = (
            current_out_dir is None
            or (self.last_auto_out_dir is not None and current_out_dir == self.last_auto_out_dir)
            or expected_out_dir.exists()
        )
        if should_auto_update:
            self.out_dir_edit.setText(str(expected_out_dir))
            self.output_dir_path = expected_out_dir
            self.last_auto_out_dir = expected_out_dir
            self.refresh_output_browser()
        self._load_day_notes(day_dir)
        self._populate_rec_choices(day_dir)
        self._refresh_cross_day_day_context(day_dir)
        self.status_label.setText(f"Loaded day {day_dir.name}")
        self._highlight_day_table_row(day_dir.name)

    def _find_day_notes_file(self, day_dir: Path) -> Optional[Path]:
        exact = sorted(day_dir.glob(f"{day_dir.name}_*.md"))
        if exact:
            return exact[0]
        any_md = sorted(day_dir.glob("*.md"))
        return any_md[0] if any_md else None

    def _load_day_notes(self, day_dir: Path) -> None:
        notes_path = self._find_day_notes_file(day_dir)
        if notes_path is None:
            self.notes_path_label.setText("No day notes detected")
            self.notes_view.setMarkdown("")
            return
        self.notes_path_label.setText(f"Notes file: {notes_path.name}")
        try:
            text = notes_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = notes_path.read_text(errors="replace")
        self.notes_view.setMarkdown(text)

    def _populate_rec_choices(self, day_dir: Path) -> None:
        validation_current = self.validation_rec_combo.currentText().strip()
        trial_current = self.trial_rec_combo.currentText().strip()
        replay_current = self.replay_rec_combo.currentText().strip()
        recs = sorted(p.name for p in day_dir.iterdir() if p.is_dir() and p.name.isdigit())
        for combo, current in (
            (self.validation_rec_combo, validation_current),
            (self.trial_rec_combo, trial_current),
            (self.replay_rec_combo, replay_current),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(recs)
            if current and current in recs:
                combo.setCurrentText(current)
            elif recs:
                combo.setCurrentIndex(0)
            else:
                combo.setEditText("001")
            combo.blockSignals(False)

    def _refresh_cross_day_day_context(self, day_dir: Path) -> None:
        recs = sorted(p.name for p in day_dir.iterdir() if p.is_dir() and p.name.isdigit())
        if recs:
            self.cross_day_available_recs_label.setText("Available recs: " + ", ".join(recs))
        else:
            self.cross_day_available_recs_label.setText("Available recs: none")

        expected_out_dir = day_dir.parent / "claude" / "figures" / "cross_day_beh"
        current_out_dir = self.cross_day_out_dir_edit.text().strip()
        if not current_out_dir or (self.last_auto_cross_day_out_dir is not None and current_out_dir == str(self.last_auto_cross_day_out_dir)):
            self.cross_day_out_dir_edit.setText(str(expected_out_dir))
            self.last_auto_cross_day_out_dir = expected_out_dir
        self._load_cross_day_selections_from_csv()

    def _cross_day_out_dir_path(self) -> Optional[Path]:
        text = self.cross_day_out_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).resolve()

    def _cross_day_csv_path(self) -> Optional[Path]:
        out_dir = self._cross_day_out_dir_path()
        if out_dir is None:
            return None
        return out_dir / "cross_day_summary_metrics.csv"

    def _load_cross_day_selections_from_csv(self) -> None:
        csv_path = self._cross_day_csv_path()
        self.cross_day_selection_list.clear()
        if csv_path is None or not csv_path.exists():
            return
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    day = str(row.get("day", "")).strip()
                    rec_tokens = [token.strip() for token in str(row.get("selected_recs", "")).split(",") if token.strip()]
                    if not day:
                        continue
                    item_text = f"{day}: {','.join(rec_tokens)}" if rec_tokens else day
                    self.cross_day_selection_list.addItem(item_text)
        except Exception as exc:
            self._append_log(f"Could not load cross-day CSV {csv_path}: {exc}")

    def _day_dir_path(self) -> Optional[Path]:
        text = self.day_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).resolve()

    def _out_dir_path(self) -> Optional[Path]:
        text = self.out_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).resolve()

    def _repo_root_and_day(self) -> tuple[Path, str]:
        day_dir = self._day_dir_path()
        if day_dir is None:
            raise ValueError("Choose a day directory first.")
        return day_dir.parent, day_dir.name

    def _selected_rec(self) -> str:
        rec = self.validation_rec_combo.currentText().strip()
        if not rec:
            raise ValueError("Choose a recording.")
        return rec

    def _selected_trial_rec(self) -> str:
        rec = self.trial_rec_combo.currentText().strip()
        if not rec:
            raise ValueError("Choose a recording.")
        return rec

    def _parse_int_csv(self, text: str) -> list[int]:
        out: list[int] = []
        for token in [part.strip() for part in text.replace(" ", ",").split(",") if part.strip()]:
            out.append(int(token))
        return out

    def _parse_str_csv(self, text: str) -> list[str]:
        return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]

    def _parse_rec_spec_text(self, text: str) -> list[int]:
        cleaned = text.strip()
        if not cleaned:
            return []
        out: list[int] = []
        for token in [part.strip() for part in cleaned.split(",") if part.strip()]:
            if "-" in token:
                start_s, end_s = token.split("-", 1)
                start_i = int(start_s)
                end_i = int(end_s)
                step = 1 if end_i >= start_i else -1
                out.extend(list(range(start_i, end_i + step, step)))
            else:
                out.append(int(token))
        deduped: list[int] = []
        for rec in out:
            if rec not in deduped:
                deduped.append(rec)
        return deduped

    def _append_log(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.log_view.append(text)

    def _set_busy(self, message: str) -> None:
        self.status_label.setText(message)

    def _set_ready(self, message: str = "Ready") -> None:
        self.status_label.setText(message)

    def _run_worker(self, message: str, fn: Callable[..., Any], *args: Any, on_done: Optional[Callable[[Any], None]] = None, **kwargs: Any) -> None:
        self._set_busy(message)
        worker = Worker(fn, *args, **kwargs)
        worker.signals.finished.connect(lambda result, logs: self._worker_finished(result, logs, on_done))
        worker.signals.error.connect(self._worker_failed)
        self.thread_pool.start(worker)

    def _worker_finished(self, result: Any, logs: str, on_done: Optional[Callable[[Any], None]]) -> None:
        self._append_log(logs)
        if on_done is not None:
            on_done(result)
        self._set_ready()

    def _worker_failed(self, error_text: str) -> None:
        self._append_log(error_text)
        self._set_ready("Error")
        QMessageBox.critical(self, "proc_gui", error_text)

    def _show_selected_image(self) -> None:
        items = self.image_list.selectedItems()
        if not items:
            return
        image_path = Path(items[0].data(Qt.ItemDataRole.UserRole))
        self.current_image = image_path
        self.preview.set_image(image_path)

    def refresh_output_browser(self) -> None:
        out_dir = self._out_dir_path()
        self.output_dir_path = out_dir
        self.image_list.clear()
        if out_dir is None or not out_dir.exists():
            self.preview.clear_image("Output directory does not exist yet")
            return
        images = sorted([p for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
        for image_path in images:
            item = QListWidgetItem(image_path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(image_path))
            self.image_list.addItem(item)
        if images:
            self.image_list.setCurrentRow(0)
        else:
            self.preview.clear_image("No images found in output directory")

    def _set_output_browser_dir(self, out_dir: Path) -> None:
        self.out_dir_edit.setText(str(out_dir))
        self.output_dir_path = out_dir
        self.refresh_output_browser()

    def add_cross_day_selection(self) -> None:
        day_dir = self._day_dir_path()
        if day_dir is None:
            QMessageBox.warning(self, "proc_gui", "Choose a day directory first.")
            return
        recs = self._parse_rec_spec_text(self.cross_day_recs_edit.text())
        if recs:
            rec_text = ",".join(f"{rec:03d}" for rec in recs)
            item_text = f"{day_dir.name}: {rec_text}"
        else:
            item_text = f"{day_dir.name}"

        existing_row = None
        for i in range(self.cross_day_selection_list.count()):
            current_text = self.cross_day_selection_list.item(i).text()
            current_day = current_text.split(":", 1)[0].strip()
            if current_day == day_dir.name:
                existing_row = i
                break
        if existing_row is None:
            self.cross_day_selection_list.addItem(item_text)
        else:
            self.cross_day_selection_list.item(existing_row).setText(item_text)

    def remove_cross_day_selection(self) -> None:
        for item in list(self.cross_day_selection_list.selectedItems()):
            row = self.cross_day_selection_list.row(item)
            self.cross_day_selection_list.takeItem(row)

    def clear_cross_day_selections(self) -> None:
        self.cross_day_selection_list.clear()

    def generate_day_summary(self) -> None:
        repo_root, day = self._repo_root_and_day()
        out_dir = self._out_dir_path()
        exclude_recs = self._parse_int_csv(self.summary_exclude_recs_edit.text())
        task_types = self._parse_str_csv(self.summary_task_types_edit.text())

        def _done(result: Any) -> None:
            if isinstance(result, dict):
                self.out_dir_edit.setText(str(result.get("out_dir", out_dir)))
            self.refresh_output_browser()

        self._run_worker(
            "Generating day summary...",
            generate_day_presentation_plots,
            repo_root,
            day,
            out_dir,
            exclude_recs,
            task_types,
            on_done=_done,
        )

    def generate_cross_day_summary(self) -> None:
        day_dir = self._day_dir_path()
        if day_dir is None:
            QMessageBox.warning(self, "proc_gui", "Choose a day directory first.")
            return
        selections = [self.cross_day_selection_list.item(i).text() for i in range(self.cross_day_selection_list.count())]
        if not selections:
            QMessageBox.warning(self, "proc_gui", "Add at least one day to the cross-day selection list.")
            return
        out_dir_text = self.cross_day_out_dir_edit.text().strip()
        out_dir = Path(out_dir_text).resolve() if out_dir_text else day_dir.parent / "claude" / "figures" / "cross_day_beh"
        task_types = self._parse_str_csv(self.cross_day_task_types_edit.text())

        def _done(result: Any) -> None:
            self._set_output_browser_dir(out_dir)
            self._load_cross_day_selections_from_csv()
            if isinstance(result, dict):
                self._append_log(json_like(result))

        self._run_worker(
            "Generating cross-day summary...",
            generate_cross_day_plots,
            day_dir.parent,
            selections,
            out_dir,
            task_types,
            on_done=_done,
        )

    def run_processing_pipeline(self) -> None:
        day_dir = self._day_dir_path()
        if day_dir is None:
            QMessageBox.warning(self, "proc_gui", "Choose a day directory first.")
            return
        rec_text = self.pipeline_rec_edit.text().strip() or None

        step_map = {
            "extract": self.step_extract_checkbox,
            "events_nodisplay": self.step_events_nodisplay_checkbox,
            "detect_display": self.step_detect_display_checkbox,
            "full_proc": self.step_full_proc_checkbox,
            "save_trials": self.step_save_trials_checkbox,
        }
        steps = {name for name, cb in step_map.items() if cb.isChecked()} or None

        def _done(result: Any) -> None:
            self._sync_day_dir_fields()
            self._refresh_day_table()
            if isinstance(result, dict):
                self._append_log(json_like(result))

        self._run_worker(
            "Running day pipeline...",
            run_day_pipeline,
            day_dir,
            skip_video=self.skip_video_checkbox.isChecked(),
            skip_existing_video=self.skip_existing_video_checkbox.isChecked(),
            rec=rec_text,
            steps=steps,
            on_done=_done,
        )

    def generate_validation_report(self) -> None:
        repo_root, day = self._repo_root_and_day()
        out_dir = self._out_dir_path()
        rec = self._selected_rec()
        sample_text = self.validation_sample_trials_edit.text().strip()
        sample_trials = parse_trial_tokens(sample_text.split()) if sample_text else None

        def _done(result: Any) -> None:
            self.refresh_output_browser()
            if isinstance(result, dict):
                self._append_log(json_like(result))

        self._run_worker(
            "Generating validation report...",
            build_validation_report,
            repo_root,
            day,
            rec,
            out_dir,
            sample_trials,
            on_done=_done,
        )

    def generate_selected_trial_plot(self) -> None:
        try:
            repo_root, day = self._repo_root_and_day()
            out_dir = self._out_dir_path()
            rec = self._selected_trial_rec()
        except Exception as exc:
            QMessageBox.warning(self, "proc_gui", str(exc))
            return
        if out_dir is None:
            QMessageBox.warning(self, "proc_gui", "Choose an output directory.")
            return
        trial = int(self.trial_spin.value())
        plot_kind = self.plot_kind_combo.currentText()

        def _make_plot() -> str:
            dataset = load_joystick_dataset(repo_root, day, rec)
            trial_index = trial - 1
            segment = get_trial_segment(dataset, trial_index)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"trial_{trial:03d}_{plot_kind}.png"
            if plot_kind == "trajectory":
                plot_trial_trajectory(segment, out_path)
            else:
                plot_trial_timeseries(segment, out_path)
            print(f"Saved {out_path}")
            return str(out_path)

        def _done(result: Any) -> None:
            self.refresh_output_browser()
            if isinstance(result, str):
                target = Path(result)
                for i in range(self.image_list.count()):
                    item = self.image_list.item(i)
                    if Path(item.data(Qt.ItemDataRole.UserRole)) == target:
                        self.image_list.setCurrentItem(item)
                        break

        self._run_worker(
            f"Generating {plot_kind} plot for trial {trial}...",
            _make_plot,
            on_done=_done,
        )

    def _schedule_auto_trial_plot(self) -> None:
        if self.auto_refresh_checkbox.isChecked():
            self.auto_plot_timer.start()

    def render_trial_replay(self) -> None:
        try:
            repo_root, day = self._repo_root_and_day()
        except Exception as exc:
            QMessageBox.warning(self, "proc_gui", str(exc))
            return
        rec = self.replay_rec_combo.currentText().strip()
        if not rec:
            QMessageBox.warning(self, "proc_gui", "Choose a recording.")
            return
        fps = int(self.replay_fps_spin.value())
        speed = float(self.replay_speed_spin.value())

        mode = self._replay_mode_group.checkedId()
        if mode == 0:
            trials_text = self.replay_trials_edit.text().strip()
            trial_numbers = parse_trial_tokens(trials_text.split()) if trials_text else [1]
            kwargs: dict = dict(trial_numbers=trial_numbers)
        else:
            kwargs = dict(
                t_start_rec_s=float(self.replay_start_spin.value()),
                duration_s=float(self.replay_duration_spin.value()),
            )

        self.replay_play_btn.setEnabled(False)
        self.replay_stop_btn.setEnabled(False)
        self.replay_replay_btn.setEnabled(False)
        self.replay_download_btn.setEnabled(False)
        self.replay_frame_info_label.setText("Rendering joystick overlay…")
        self.replay_display.clear_frame("Rendering…")
        self.replay_cam_top.clear_frame("Loading…")
        self.replay_cam_left.clear_frame("Loading…")
        self._replay_timer.stop()
        if self._replay_session is not None:
            self._replay_session.close()
            self._replay_session = None

        def _done(session: Any) -> None:
            self._on_replay_session_ready(session, fps)

        self._run_worker(
            "Rendering replay...",
            build_replay_session,
            repo_root, day, rec,
            fps=fps, playback_speed=speed,
            on_done=_done,
            **kwargs,
        )

    def _on_replay_session_ready(self, session: Any, fps: int) -> None:
        if session is None or not session.mpl_frames:
            self.replay_frame_info_label.setText("No frames rendered.")
            self.replay_display.clear_frame("No frames returned.")
            return
        self._replay_session = session
        self._replay_frame_idx = 0
        self._replay_fps = fps

        if not session.cameras:
            self.replay_cam_top.clear_frame("No camera data")
            self.replay_cam_left.clear_frame("No camera data")
        else:
            if "Cam Top" not in session.cameras:
                self.replay_cam_top.clear_frame("Cam Top not found")
            if "Cam Left" not in session.cameras:
                self.replay_cam_left.clear_frame("Cam Left not found")

        self.replay_download_btn.setEnabled(True)
        self.replay_replay_btn.setEnabled(True)
        self._replay_show_frame(0)
        self._replay_play()

    def _replay_show_frame(self, idx: int) -> None:
        session = self._replay_session
        if session is None or idx >= len(session.mpl_frames):
            return
        pixmap = QPixmap()
        pixmap.loadFromData(session.mpl_frames[idx])
        self.replay_display.set_frame(pixmap)

        t_now = float(session.frame_times[idx])
        for cam_name, display in (("Cam Top", self.replay_cam_top), ("Cam Left", self.replay_cam_left)):
            reader = session.cameras.get(cam_name)
            if reader is None:
                continue
            frame = reader.read_at_perf(t_now)
            if frame is not None:
                display.set_frame(_numpy_bgr_to_pixmap(frame))

        n = len(session.mpl_frames)
        self.replay_frame_info_label.setText(f"Frame {idx + 1} / {n}  t={t_now:.2f} s")

    def _replay_play(self) -> None:
        if self._replay_session is None:
            return
        self.replay_play_btn.setEnabled(False)
        self.replay_stop_btn.setEnabled(True)
        self._replay_timer.start(max(1, int(1000 / self._replay_fps)))

    def _replay_stop(self) -> None:
        self._replay_timer.stop()
        self.replay_play_btn.setEnabled(self._replay_session is not None)
        self.replay_stop_btn.setEnabled(False)

    def _replay_replay(self) -> None:
        self._replay_timer.stop()
        self._replay_frame_idx = 0
        self._replay_show_frame(0)
        self._replay_play()

    def _replay_tick(self) -> None:
        session = self._replay_session
        if session is None:
            self._replay_timer.stop()
            return
        self._replay_frame_idx += 1
        if self._replay_frame_idx >= len(session.mpl_frames):
            self._replay_timer.stop()
            self._replay_frame_idx = len(session.mpl_frames) - 1
            self.replay_play_btn.setEnabled(True)
            self.replay_stop_btn.setEnabled(False)
            return
        self._replay_show_frame(self._replay_frame_idx)

    def _replay_download(self) -> None:
        try:
            repo_root, day = self._repo_root_and_day()
        except Exception as exc:
            QMessageBox.warning(self, "proc_gui", str(exc))
            return
        rec = self.replay_rec_combo.currentText().strip()
        if not rec:
            QMessageBox.warning(self, "proc_gui", "Choose a recording.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Replay MP4", f"replay_{day}_{rec}.mp4", "MP4 Video (*.mp4)"
        )
        if not out_path:
            return
        trials_text = self.replay_trials_edit.text().strip()
        trial_numbers = parse_trial_tokens(trials_text.split()) if trials_text else [1]
        fps = int(self.replay_fps_spin.value())
        speed = float(self.replay_speed_spin.value())

        def _done(result: Any) -> None:
            if isinstance(result, Path):
                self._append_log(f"Saved replay to {result}")

        self._run_worker(
            "Saving MP4...",
            render_trial_replay_video,
            repo_root, day, rec, trial_numbers, out_path,
            fps=fps, playback_speed=speed,
            on_done=_done,
        )

    def closeEvent(self, event: Any) -> None:
        if self._replay_session is not None:
            self._replay_session.close()
            self._replay_session = None
        s = QSettings("PesaranLab", "ProcGui")
        if self._monkey_dir is not None:
            s.setValue("ProcGui/monkey_dir", str(self._monkey_dir))
        else:
            s.remove("ProcGui/monkey_dir")
        super().closeEvent(event)


def json_like(value: Any) -> str:
    try:
        return json.dumps(value, indent=2)
    except Exception:
        return str(value)


def run() -> None:
    app = QApplication([])
    window = ProcGuiWindow()
    window.showMaximized()
    app.exec()
