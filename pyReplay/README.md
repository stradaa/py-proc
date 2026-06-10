# pyReplay — replaying thalamus behave files

`pyReplay` is a standalone tool for **re-playing a time window of a thalamus
behave recording** as synchronized panels: every camera, the joystick cursor +
task target, and the analog NIDAQ channels (fiducial / photodiode / reward),
all driven from the same clock and viewable in a scrubbable PyQt6 window or
exported to a single composite MP4.

This README is the **canonical, recreate-from-scratch reference** for working
with behave files: the data model, the timestamp rules, how each stream is
parsed, and where the load-bearing knowledge came from. If you only read one
section, read **[The one rule that matters](#the-one-rule-that-matters-timestamps)**.

---

## 0. Environment

Run everything with the **py-proc venv** (Python 3.10) — it has `thalamus`,
`PyQt6`, `cv2` (opencv), `matplotlib`, `numpy`, `scipy`:

```
/vol/cortex/nvme-envs/pesaranlab/alex/py-proc/bin/python
```

- The thalamus "main" env (`/vol/cortex/nvme-envs/pesaranlab/alex/main`, py3.11)
  has `thalamus` but **no PyQt6**, so it can't run the GUI.
- The *system* `python3` fails importing `thalamus_pb2` (protobuf permission error).

A behave file viewer for quick inspection (prints every record):

```
python -m thalamus.record_reader2 /cdz/pesaranlab/Eevee_Behavior_AlexRig/260608/behave.20260608.1
```

---

## 1. What a behave file is

A recording lives in a day directory, e.g.
`/cdz/pesaranlab/Eevee_Behavior_AlexRig/260608/`, with one capture per run:

```
behave.20260608.1            # the capture file (length-prefixed protobuf stream)
behave.20260608.1.json       # node configuration for that run
behave.20260608.1.Cam Top.avi    # per-camera MPEG4 sidecars (see §3 — we DON'T use these for sync)
behave.20260608.1.Cam Left.avi
behave.20260608.1.Cam Rear.avi
```

The capture file is a flat stream of **`StorageRecord`** protobuf messages, each
length-prefixed by an 8-byte big-endian size. Every record carries:

- **`node`** — the producing node's name (a string, e.g. `"Cam Top"`, `"Joystick"`, `"Analog in"`).
- **`time`** — a **nanosecond timestamp on a shared monotonic clock** (see §2).
- a **`body`** oneof — exactly one of: `analog`, `image`, `text`, `compressed`,
  plus a few we ignore (`xsens`, `metadata`).

Read it with `thalamus.record_reader2.RecordReader`:

```python
from thalamus.record_reader2 import RecordReader

with RecordReader(path, decode_video=True) as reader:   # signature: (file, node=None, decompress=True, decode_video=True)
    for record in reader:
        t_ns = record.time
        body = record.WhichOneof("body")
        if body == "image":
            img = record.image      # decode_video=True -> img.data[0] is decoded Gray pixels
        elif body == "analog":
            a = record.analog       # a.spans (named channels), a.data, a.sample_intervals, a.time
        elif body == "text":
            txt = record.text.text  # behavioral state strings + per-trial JSON summaries
```

Key facts about `RecordReader`:

- **No seeking.** It reads linearly from the start. But records are written in
  time order, so to load a window `[start, stop]` you iterate and **`break` as
  soon as `record.time > stop`**. A late window still reads from 0 (cost scales
  with `stop`, not window length).
- **`decode_video`**: with `True`, MPEG4 `image` records are decoded through
  ffmpeg and `image.data[0]` is raw `Gray` bytes (`width*height`). With `False`,
  it's hundreds of times faster but the image bytes are opaque. **We use `True`**
  so each frame carries its own timestamp (see §3).
- The class spins up an ffmpeg subprocess per camera; non-image records still
  flow through normally while video decodes in the background.

---

## 2. The one rule that matters: timestamps

> **`record.time` (nanoseconds) is the ONLY authority for time. It is shared
> across every node. Ignore every framerate / sampling-rate field in the
> metadata — they are wrong.**

To synchronize streams you place each datum at its `record.time` and read off
"what is each stream showing at time *t*". Concretely in this rig:

- Cameras report `image.frame_interval` and the `.avi` sidecars report an fps —
  **both are unreliable** (Cam Top.avi claims 120 fps; the real per-frame
  spacing from `record.time` is ~16.7 ms ⇒ 60 fps). Always derive timing from
  consecutive `record.time` values.
- The task's per-trial JSON events carry `time_perf_counter` (seconds). On this
  single-machine rig **that perf-counter clock IS the same monotonic clock as
  `record.time`**, so `perf * 1e9` is directly a behave-clock nanosecond.
  Verified: a trial's `target_on` `perf*1e9 − t0_ns` lands exactly on the
  `start_on` state's `record.time`. This is what lets us place task events with
  no cross-clock fitting.

`t0_ns` = the first record's `time`. A window selected as "30–60 s" means
`[t0_ns + 30e9, t0_ns + 60e9]`. Relative seconds = `(record.time − t0_ns)/1e9`.

---

## 3. Cameras

**Source of truth = the behave file, NOT the `.avi` sidecars.** The cameras are
stored both inline (MPEG4 `image` records) and as `.avi` sidecars. The early
version of this tool seeked the `.avi` by frame index and paired frames with
behave timestamps — that desynced, because it relied on frame-count/fps
metadata. The fix:

- Read with `RecordReader(decode_video=True)`. Each `image` record then yields a
  decoded `Gray` frame in `image.data[0]` **paired with its own `record.time`**.
- Reshape to `(height, width)` uint8 and downsize to a target height to bound
  memory (`cameras.py: frame_from_image`).
- Cameras run at **different rates** (Top 60 fps, Left/Rear 40 fps) — irrelevant,
  because each frame is placed by its timestamp and looked up by nearest-time.

Memory note: frames are held in RAM for smooth playback, so window size × number
of cameras × resolution is bounded by `target_h` (default 360 px).

---

## 4. Joystick → cursor, and the task target

### Joystick samples
The `Joystick` node emits `analog` records with spans named `X`, `Y` (and
`Frequency`). Each packet holds several samples; one `record.time`. Extraction
mirrors `py_proc/procThalamus_indie.py:_extract_joystick_samples`
(`loader.py:_joystick_samples`): values come from the `X`/`Y` spans; per-sample
timestamps end at `record.time` and step back by `sample_intervals` **only to
spread samples *within* a packet** — the cross-stream anchor is always
`record.time`.

### Cursor reconstruction (normalized [0,1] task space)
Raw joystick deflection is in `[-1, 1]`; the task target is in normalized
`[0,1]` task coordinates. To overlay them they must be in the same space, so we
reconstruct the **cursor** exactly as the task does (`cursor.py`, mirroring
`pyCheck/joystick_validation.py:reconstruct_cursor`):

- `control_mode = "direct"` (this session, `direct_range=0.45`):
  `cursor = 0.5 + influence(joystick) * direct_range` — stateless.
- cumulative / velocity mode: integrate joystick over time, resetting to the
  center `(0.5, 0.5)` at each trial start.

`apply_direction_influence` scales each quadrant by `{up,down,left,right}_influence_pct`.

### Where the target comes from — and precise on/off timing
Every trial writes a large (~40 KB) JSON **`text`** record at the **trial's
end**, with top-level keys `used_values / task_config / task_result /
behav_result`:

- `task_config.targets[]` — the menu of possible targets (`x_norm`, `y_norm`,
  `radius_ratio`, `enabled`, ...). `task_config.center_gate_radius_ratio` is the
  center gate.
- `behav_result.attempts[]` (the last == `final_attempt`) — what actually
  happened. Each attempt has an **`events`** list; for replay we use:
  - one **`target_on`** event (carries `target_x`, `target_y`,
    `target_radius_ratio`, `time_perf_counter`),
  - a resolving event: **`success`** or **`*_fail`** (e.g. `touch_input_fail`).

The target is shown **only over `[target_on, resolution]`** — converting
`time_perf_counter * 1e9` to behave ns (§2). This is why the target blinks on at
trial start and off at the outcome, and stays hidden during the intertrial gap.
(`loader.py:_summary_targets`.)

The behave file *also* emits coarse `BehavState=` text records on the behave
clock — `start_on` → `success`/`fail` → `intertrial` — but they're emitted
~0.13 s after the actual event, so we prefer the event `time_perf_counter`.

---

## 5. Analog NIDAQ channels

The `Analog in` node (NIDAQ) runs at **1000 Hz** with three channels mapped
(AlexRig, from `procThalamus_indie.py`):

| span name   | meaning              |
|-------------|----------------------|
| `Dev1/ai0`  | fiducial             |
| `Dev1/ai1`  | photodiode / display |
| `Dev1/ai2`  | reward (analog)      |

Each `analog` record holds many samples per channel ending at `record.time`;
per-sample timestamps are reconstructed the same way as the joystick
(`loader.py:_sample_times_ns`). Other analog-ish nodes exist (`Reward`,
`Fiducial`, `Oculomatic`, `Node 2` touchscreen) — see `procThalamus_indie.py`.

---

## 6. Package layout

All files are kept **under 500 lines** (project rule: split before the limit).

| file          | role |
|---------------|------|
| `window.py`   | `ReplayWindow` data model + time helpers (`rel_to_ns`, `active_target`, nearest-index lookups). `CameraTrack`, `AnalogTrack`, `EventMark`, `TargetSpec`. |
| `loader.py`   | One `decode_video=True` pass → `ReplayWindow`. Parses cameras, joystick, analog, `BehavState` events, and target intervals; reconstructs the cursor. Breaks at `stop`. |
| `cameras.py`  | `frame_from_image` — reshape + downsize one decoded Gray frame. |
| `cursor.py`   | joystick → cursor (`[0,1]`) reconstruction (direct + cumulative). |
| `panels.py`   | `SignalPanels`: matplotlib joystick-2D (cursor + target + center gate) and scrolling analog traces (±`scroll_s`, fixed central playhead). Can build into an external figure (used by export). |
| `blit.py`     | `BlitManager` — fast partial redraws for smooth live playback. |
| `player.py`   | `ReplayPlayer` — PyQt6 window: cameras across the top (QPixmap), panels below (embedded blitted matplotlib), play/pause/seek/speed/scroll + **Export MP4**. |
| `export.py`   | `export_mp4` — one figure (camera row via `imshow` + `SignalPanels`) driven by `FFMpegWriter`, real-time by default. |
| `run_replay.py` / `__main__.py` | CLI entry. |

### Data flow
`load_window()` → `ReplayWindow` (cameras as decoded frame lists + timestamps,
joystick/cursor arrays, analog tracks, target intervals). The player/export then
ask, for each output time *t*: nearest camera frame per cam, cursor sample +
trail, active target, and the ±`scroll_s` slice of each analog channel.

---

## 7. Running it

```bash
cd /cdz/pesaranlab/agents/alex/py-proc
PY=/vol/cortex/nvme-envs/pesaranlab/alex/py-proc/bin/python

# Interactive player, 30–60 s of a recording:
$PY -m pyReplay --file /cdz/pesaranlab/Eevee_Behavior_AlexRig/260608/behave.20260608.1 --start 30 --stop 60
# or address by day dir + rec number:
$PY -m pyReplay --day-dir /cdz/pesaranlab/Eevee_Behavior_AlexRig/260608 --rec 1 --start 30 --stop 60
```

Options: `--cameras "Cam Top" "Cam Left"` (default auto-detect all), `--scroll`
(trace half-window, default 2 s), `--target-h` (decoded frame height, default
360). The window has **Export MP4** (real-time composite) and live
play/pause/seek/speed/scroll controls.

Programmatic use:

```python
from pyReplay.loader import load_window
from pyReplay.export import export_mp4
win = load_window(behave_path, 30, 60)     # preload (decodes 0→stop once)
print(win.summary())
export_mp4(win, "out.mp4", scroll_s=2.0, fps=30)
```

---

## 8. Reference code that made this possible

- **`thalamus` package** — `…/main/lib/python3.11/site-packages/thalamus/`:
  - `record_reader2.py` — `RecordReader` (the reader + ffmpeg video decode path).
  - `dataframe.py` — `DataFrameBuilder` (analog/text → DataFrame; per-sample timing semantics).
  - `video_writer.py` — `MultiVideoWriter` / `VideoWriter` (how thalamus muxes video).
  - `thalamus_pb2` — `StorageRecord` / `Image` / `Analog` proto definitions (`Image.Format`: Gray=0 … MPEG1=7, MPEG4=8).
- **Jarl Haggerty's sample** (posted in Slack) — the original `RecordReader` +
  `DataFrameBuilder` + `MultiVideoWriter` example that established how to iterate
  records, build per-node DataFrames keyed by the nanosecond `count` index, and
  merge them with `pandas.merge_asof`. The basis for §1–§2.
- **`py_proc/procThalamus_indie.py`** — the production parser: AlexRig channel
  mapping (`Dev1/ai0/1/2`), `_extract_joystick_samples`, node names, wallclock
  handling. Our `loader.py` deliberately mirrors it for consistency.
- **`pyCheck/joystick_validation.py`** — the joystick aesthetic (`plot_trial_trajectory`),
  `reconstruct_cursor` / `apply_direction_influence`, and `CameraReader`
  (the earlier `.avi`-index approach we deliberately moved away from for sync).

---

## 9. Known limitations / future work

- **Window ends mid-trial:** a trial's target/summary record is emitted at the
  trial *end*. If your window's `stop` falls mid-trial, that final partial
  trial's target isn't loaded (the summary is past `stop`). Fix: scan `text`
  records a little past `stop` (cheap — no extra video decode) to grab the next
  summary.
- **Disk cache:** preloaded windows are not yet cached to disk; re-opening the
  same window re-decodes.
- **Trial/event-based window selection:** currently start/stop seconds only;
  selecting by trial number or event is a natural next step (the data is already
  parsed).
- **Performance:** load cost scales with `stop` (no seek). For deep windows an
  ffmpeg `-ss` pre-seek that still pairs frames to `record.time` would help.
