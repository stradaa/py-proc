"""Convert decoded camera image records into display-ready frames.

Frames are obtained from the behave file itself via RecordReader(decode_video=
True): each `image` record then carries a decoded grayscale frame in
image.data[0] together with its own record.time on the shared monotonic clock.
This is the only timestamping authority we use for video — no .avi sidecars, no
framerate metadata, no frame-index math, so video stays synchronized with every
other stream.

Here we just reshape and downsize one decoded frame to bound memory so a whole
window can be held in RAM for smooth playback.
"""

from __future__ import annotations

import numpy as np


def frame_from_image(image, target_h: int = 360) -> np.ndarray | None:
    """Reshape a decoded gray image record to (H, W) uint8, downsized to target_h.

    `image` is a thalamus Image proto whose data[0] holds width*height gray
    bytes (RecordReader forces Gray output when decode_video=True).
    """
    if not image.data or image.width <= 0 or image.height <= 0:
        return None
    w, h = int(image.width), int(image.height)
    buf = image.data[0]
    if len(buf) < w * h:
        return None
    frame = np.frombuffer(buf[: w * h], dtype=np.uint8).reshape(h, w)

    if target_h and h > target_h:
        import cv2

        scale = target_h / h
        frame = cv2.resize(frame, (max(1, int(round(w * scale))), target_h),
                           interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame, dtype=np.uint8)
