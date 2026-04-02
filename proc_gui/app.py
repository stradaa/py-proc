from __future__ import annotations

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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pyCheck.day_presentation_plots import generate_day_presentation_plots
from pyCheck.joystick_validation import (
    build_validation_report,
    get_trial_segment,
    load_joystick_dataset,
    parse_trial_tokens,
    plot_trial_timeseries,
    plot_trial_trajectory,
)


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

        self.rec_combo = QComboBox()
        self.rec_combo.setEditable(True)
        self.rec_combo.currentTextChanged.connect(self._schedule_auto_trial_plot)
        io_form.addRow("Rec", self.rec_combo)

        self.task_types_edit = QLineEdit()
        self.task_types_edit.setPlaceholderText("joystick_intro, other_task")
        io_form.addRow("Task Types", self.task_types_edit)

        self.exclude_recs_edit = QLineEdit()
        self.exclude_recs_edit.setPlaceholderText("3,5")
        io_form.addRow("Exclude Recs", self.exclude_recs_edit)

        self.sample_trials_edit = QLineEdit()
        self.sample_trials_edit.setPlaceholderText("1 2 4 or 90-130")
        io_form.addRow("Sample Trials", self.sample_trials_edit)

        controls_layout.addWidget(io_group)

        trial_group = QGroupBox("Trial Plot")
        trial_form = QFormLayout(trial_group)

        self.trial_spin = QSpinBox()
        self.trial_spin.setRange(1, 10000)
        self.trial_spin.valueChanged.connect(self._schedule_auto_trial_plot)
        trial_form.addRow("Trial", self.trial_spin)

        self.plot_kind_combo = QComboBox()
        self.plot_kind_combo.addItems(["timeseries", "trajectory"])
        self.plot_kind_combo.currentTextChanged.connect(self._schedule_auto_trial_plot)
        trial_form.addRow("Plot", self.plot_kind_combo)

        self.auto_refresh_checkbox = QCheckBox("Auto-refresh trial plot")
        self.auto_refresh_checkbox.setChecked(True)
        trial_form.addRow("", self.auto_refresh_checkbox)

        controls_layout.addWidget(trial_group)

        button_group = QGroupBox("Actions")
        button_layout = QVBoxLayout(button_group)

        summary_btn = QPushButton("Generate Day Summary")
        summary_btn.clicked.connect(self.generate_day_summary)
        button_layout.addWidget(summary_btn)

        validation_btn = QPushButton("Generate Validation Report")
        validation_btn.clicked.connect(self.generate_validation_report)
        button_layout.addWidget(validation_btn)

        trial_btn = QPushButton("Generate Selected Trial Plot")
        trial_btn.clicked.connect(self.generate_selected_trial_plot)
        button_layout.addWidget(trial_btn)

        refresh_btn = QPushButton("Refresh Output Browser")
        refresh_btn.clicked.connect(self.refresh_output_browser)
        button_layout.addWidget(refresh_btn)

        controls_layout.addWidget(button_group)

        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        controls_layout.addWidget(self.log_view, stretch=1)

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
        self._populate_rec_choices(day_dir)
        self.status_label.setText(f"Loaded day {day_dir.name}")

    def _populate_rec_choices(self, day_dir: Path) -> None:
        current = self.rec_combo.currentText().strip()
        recs = sorted(p.name for p in day_dir.iterdir() if p.is_dir() and p.name.isdigit())
        self.rec_combo.blockSignals(True)
        self.rec_combo.clear()
        self.rec_combo.addItems(recs)
        if current and current in recs:
            self.rec_combo.setCurrentText(current)
        elif recs:
            self.rec_combo.setCurrentIndex(0)
        else:
            self.rec_combo.setEditText("001")
        self.rec_combo.blockSignals(False)

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
        rec = self.rec_combo.currentText().strip()
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

    def generate_day_summary(self) -> None:
        repo_root, day = self._repo_root_and_day()
        out_dir = self._out_dir_path()
        exclude_recs = self._parse_int_csv(self.exclude_recs_edit.text())
        task_types = self._parse_str_csv(self.task_types_edit.text())

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

    def generate_validation_report(self) -> None:
        repo_root, day = self._repo_root_and_day()
        out_dir = self._out_dir_path()
        rec = self._selected_rec()
        sample_text = self.sample_trials_edit.text().strip()
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
            rec = self._selected_rec()
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
    window.show()
    app.exec()
