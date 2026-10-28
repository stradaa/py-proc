"""Render a ReplayWindow to a composite MP4 (cameras + cursor panel + traces).

The arrangement comes from pyReplay/layouts.py — pick one with `layout=`. The
figure is driven frame by frame with matplotlib's FFMpegWriter; output runs at
real time by default (playback_speed 1.0), stepping 1/fps * playback_speed
seconds through the window per frame.

Defaults target a share-ready file: 16:9 at 1920x1080 (figsize 16x9 at dpi 120)
encoded H.264 / yuv420p, which is what Slack, Keynote/PowerPoint and QuickTime
all play without re-encoding. Bump `dpi` to 160 for 2560x1440.
"""

from __future__ import annotations

import pathlib
from typing import Callable, Optional, Sequence, Tuple

from matplotlib.animation import FFMpegWriter

from pyReplay.layouts import DEFAULT_LAYOUT, build_layout
from pyReplay.window import ReplayWindow

# H.264 with yuv420p: the combination every player/chat client accepts. crf 18
# is visually lossless for this kind of synthetic frame at a sane file size.
FFMPEG_ARGS = ["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-preset", "medium", "-movflags", "+faststart"]

DEFAULT_FIGSIZE = (16.0, 9.0)
DEFAULT_DPI = 120                 # 16x9 in at 120 dpi = 1920x1080


def export_mp4(
    win: ReplayWindow,
    out_path: str,
    *,
    layout: str = DEFAULT_LAYOUT,
    scroll_s: float = 2.0,
    fps: int = 30,
    playback_speed: float = 1.0,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    camera_names: Optional[Sequence[str]] = None,
    start_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> str:
    """Render `win` to `out_path` and return the path.

    start_s / duration_s restrict the render to a sub-span of the window
    (seconds relative to the recording start); both default to the whole window.
    """
    comp = build_layout(layout, win, figsize=figsize, scroll_s=scroll_s,
                        camera_names=camera_names)

    t_start = win.start_s if start_s is None else max(start_s, win.start_s)
    span = win.duration_s if duration_s is None else duration_s
    span = min(span, win.stop_s - t_start)
    n_frames = max(int(span * fps / max(playback_speed, 1e-9)), 1)

    out_path = str(pathlib.Path(out_path))
    writer = FFMpegWriter(fps=fps, extra_args=FFMPEG_ARGS,
                          metadata={"comment": f"pyReplay {win.source} [{layout}]"})

    with writer.saving(comp.fig, out_path, dpi=dpi):
        for i in range(n_frames):
            comp.update(t_start + (i / fps) * playback_speed)
            writer.grab_frame()
            if progress is not None and (i % 5 == 0 or i == n_frames - 1):
                progress((i + 1) / n_frames)
    return out_path


def export_still(
    win: ReplayWindow,
    out_path: str,
    t_rel_s: float,
    *,
    layout: str = DEFAULT_LAYOUT,
    scroll_s: float = 2.0,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    camera_names: Optional[Sequence[str]] = None,
) -> str:
    """Render a single frame at `t_rel_s` to a PNG — a layout preview."""
    comp = build_layout(layout, win, figsize=figsize, scroll_s=scroll_s,
                        camera_names=camera_names)
    comp.update(t_rel_s)
    out_path = str(pathlib.Path(out_path))
    comp.fig.savefig(out_path, dpi=dpi, facecolor=comp.fig.get_facecolor())
    return out_path
