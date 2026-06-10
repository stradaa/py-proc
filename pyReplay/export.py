"""Render a ReplayWindow to a composite MP4 (cameras + joystick + traces).

Builds one matplotlib figure — a camera row across the top (imshow) above the
same SignalPanels used by the live player — and drives it frame by frame with
matplotlib's FFMpegWriter, exactly like pyCheck/joystick_validation.py's
render_trial_replay_video. Output runs at real time by default (playback_speed
1.0); frames step through the window by 1/fps * playback_speed seconds.
"""

from __future__ import annotations

import pathlib
from typing import Callable, List, Optional

import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure

from pyReplay.panels import SignalPanels
from pyReplay.window import ReplayWindow


def export_mp4(
    win: ReplayWindow,
    out_path: str,
    *,
    scroll_s: float = 2.0,
    fps: int = 30,
    playback_speed: float = 1.0,
    dpi: int = 100,
    progress: Optional[Callable[[float], None]] = None,
) -> str:
    cams = list(win.cameras)
    ncam = max(len(cams), 1)

    fig = Figure(figsize=(4.2 * ncam, 4.2 * ncam / 2 + 4.4), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.12,
                             left=0.05, right=0.97, top=0.95, bottom=0.08)

    # Camera row (imshow), one axis per camera.
    cam_gs = outer[0].subgridspec(1, ncam, wspace=0.04)
    cam_im = {}
    for col, name in enumerate(cams):
        ax = fig.add_subplot(cam_gs[0, col])
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        first = win.cameras[name].frames[0] if win.cameras[name].frames else np.zeros((2, 2), np.uint8)
        cam_im[name] = ax.imshow(first, cmap="gray", vmin=0, vmax=255,
                                 aspect="equal", animated=True)

    # Joystick + trace panels reuse the live player's SignalPanels.
    panels = SignalPanels(win, scroll_s=scroll_s, fig=fig, host=outer[1])

    n_frames = max(int(win.duration_s * fps / max(playback_speed, 1e-9)), 1)
    out_path = str(pathlib.Path(out_path))
    writer = FFMpegWriter(fps=fps, metadata={"comment": f"pyReplay {win.source}"})

    with writer.saving(fig, out_path, dpi=dpi):
        for i in range(n_frames):
            t_rel = win.start_s + (i / fps) * playback_speed
            t_ns = win.rel_to_ns(t_rel)
            for name, im in cam_im.items():
                frame = win.cameras[name].frame_at(t_ns)
                if frame is not None:
                    im.set_data(frame)
            panels.update(t_rel)
            writer.grab_frame()
            if progress is not None and (i % 5 == 0 or i == n_frames - 1):
                progress((i + 1) / n_frames)
    return out_path
