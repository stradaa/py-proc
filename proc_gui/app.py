from __future__ import annotations

import csv
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pyCheck.cross_day_plots import generate_cross_day_plots
from pyCheck.day_presentation_plots import generate_day_presentation_plots
from pyCheck.joystick_validation import (
    build_validation_report,
    get_trial_segment,
    load_joystick_dataset,
    parse_trial_tokens,
    plot_trial_timeseries,
    plot_trial_trajectory,
)
from py_proc.run_day_pipeline import run_day_pipeline


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


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

        self.no_display_checkbox = QCheckBox("No display pass only")
        processing_form.addRow("", self.no_display_checkbox)

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

        left_splitter.addWidget(inputs_container)
        left_splitter.addWidget(notes_group)
        left_splitter.addWidget(actions_container)
        left_splitter.addWidget(log_group)
        left_splitter.setStretchFactor(0, 2)
        left_splitter.setStretchFactor(1, 4)
        left_splitter.setStretchFactor(2, 5)
        left_splitter.setStretchFactor(3, 3)
        left_splitter.setSizes([130, 240, 420, 180])

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

    def _wrap_layout(self, layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _set_help_text(self, widget: QWidget, text: str) -> None:
        widget.setToolTip(text)
        widget.setStatusTip(text.replace("\n", " "))
        widget.setWhatsThis(text)
        widget.setToolTipDuration(20000)

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
        recs = sorted(p.name for p in day_dir.iterdir() if p.is_dir() and p.name.isdigit())
        for combo, current in (
            (self.validation_rec_combo, validation_current),
            (self.trial_rec_combo, trial_current),
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

        def _done(result: Any) -> None:
            self._sync_day_dir_fields()
            if isinstance(result, dict):
                self._append_log(json_like(result))

        self._run_worker(
            "Running day pipeline...",
            run_day_pipeline,
            day_dir,
            self.skip_video_checkbox.isChecked(),
            self.skip_existing_video_checkbox.isChecked(),
            self.no_display_checkbox.isChecked(),
            rec_text,
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


def json_like(value: Any) -> str:
    import json

    try:
        return json.dumps(value, indent=2)
    except Exception:
        return str(value)


def run() -> None:
    app = QApplication([])
    window = ProcGuiWindow()
    window.showMaximized()
    app.exec()
