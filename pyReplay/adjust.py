"""Per-camera brightness / contrast / gamma correction for replay frames.

Replay MP4s (and the live player) can look very dim on some cameras while others
are fine, so each camera gets its own correction. The correction is a pure
intensity remap encoded as a 256-entry lookup table (LUT): applying it to a
grayscale uint8 frame is just `lut[frame]` — a cheap gather with no per-pixel
float math — so it is free enough to run every frame in both the live player and
the MP4 exporter.

The remap (per camera, on the 0..255 range) is gamma first, then a linear
gain/bias:

    g   = (i / 255) ** (1 / gamma)      # gamma > 1 lifts dark midtones
    out = clip(gain * (g * 255) + bias, 0, 255)

`gain=1, bias=0, gamma=1` is the identity (frames pass through untouched).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CameraAdjust:
    """Intensity correction for one camera.

    gain  : contrast multiplier (1.0 = unchanged).
    bias  : brightness offset added after gain, in 0..255 units.
    gamma : midtone lift; >1 brightens shadows, <1 darkens (1.0 = unchanged).
    """

    gain: float = 1.0
    bias: float = 0.0
    gamma: float = 1.0
    _lut: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def is_identity(self) -> bool:
        return self.gain == 1.0 and self.bias == 0.0 and self.gamma == 1.0

    def lut(self) -> np.ndarray:
        """Cached 256-entry uint8 lookup table for this correction."""
        if self._lut is None:
            i = np.arange(256, dtype=np.float64)
            g = np.power(i / 255.0, 1.0 / self.gamma) if self.gamma != 1.0 else i / 255.0
            out = self.gain * (g * 255.0) + self.bias
            self._lut = np.clip(out, 0, 255).astype(np.uint8)
        return self._lut

    def apply(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return the corrected frame (or the frame unchanged if identity/None)."""
        if frame is None or self.is_identity:
            return frame
        return self.lut()[frame]

    def ffmpeg_eq(self) -> Optional[str]:
        """This correction as an ffmpeg `eq` filter string, or None if identity.

        Used only by the fallback re-encode path in py_proc/video_remux.py (when
        the exact LUT re-encode is unavailable). For the gamma-only corrections we
        use, ffmpeg's `eq` gamma matches the LUT exactly (both are
        pow(i/255, 1/gamma)*255); gain/bias map approximately onto eq's
        contrast/brightness, which is acceptable for a viewing-only fallback.
        """
        if self.is_identity:
            return None
        parts = [f"gamma={self.gamma:g}"]
        if self.gain != 1.0:
            parts.append(f"contrast={self.gain:g}")
        if self.bias != 0.0:
            parts.append(f"brightness={self.bias / 255.0:g}")
        return "eq=" + ":".join(parts)


# Per-camera defaults, keyed by behave node name. Cam Eye is bright and left at
# identity; the other three are dim and get a gamma lift sized to how dark each
# is (Cam Rear is darkest). Tuned interactively via pyReplay/tune_contrast.py on
# 260715 rec 001 (raw means: Eye 217, Left 65, Top 51, Rear 34).
DEFAULTS: dict[str, CameraAdjust] = {
    "Cam Eye": CameraAdjust(),
    "Cam Top": CameraAdjust(gamma=2.0),
    "Cam Left": CameraAdjust(gamma=1.8),
    "Cam Rear": CameraAdjust(gamma=2.2),
}


def resolve(name: str) -> CameraAdjust:
    """Correction for camera `name`; identity (pass-through) if unknown."""
    return DEFAULTS.get(name, CameraAdjust())
