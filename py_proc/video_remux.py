"""Reconstruct per-camera MP4s from a thalamus behave file's own frames.

The behave file stores each camera frame as one MPEG4 ``image`` record carrying
its own ``record.time`` (ns) on the shared monotonic clock. ``record.time`` is
the *only* valid timing authority — framerate / sample-rate metadata is wrong
(see ``pyReplay/README.md`` §2-3: a sidecar may claim 120 fps while the real
per-frame spacing from ``record.time`` is 60 fps). This module rebuilds
``rec{N}.bag/img/{node}/image.mp4`` from those frames, one output frame per
valid image record, so the video is **1:1** with the
``{node}_image_raw_compressed.mat`` timestamps the rest of the pipeline writes.
A 1:1 rebuild makes ``pyCheck/joystick_validation.py:CameraReader``'s
``avi_frames / n_mat`` scaling exactly 1.0 — frame ``i`` is timestamp ``i``.

Two encode paths, auto-selected by :func:`make_camera_video`:

* :func:`_remux_pyav` — *lossless*: re-mux the original MPEG4 packets, stamping
  each with a presentation timestamp derived from ``record.time`` (variable
  frame rate, true PTS, bit-exact pixels). Requires PyAV (``av``).
* :func:`_reencode_ffmpeg` — fallback when PyAV is unavailable: decode the
  contiguous MPEG4 elementary stream and re-encode 1:1 with libx264, preserving
  every frame. The ``.mat`` timestamps remain the timing authority.

The frame-validity guard mirrors ``pyReplay/cameras.py:frame_from_image``: an
``image`` record is a real frame only when ``width > 0`` and it actually carries
bytes. The trailing ``time=0, width=0, data=[b'']`` sentinel record is dropped
by the caller before packets reach this module.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from fractions import Fraction
from typing import Iterable, Sequence, Tuple

# Time base for the lossless remux: 90 kHz ticks (standard MPEG fine clock).
# Do NOT use 1/1e9 — nanosecond tick counts overflow 32-bit PTS fields on long
# sessions and several muxers reject the time base.
_TIME_BASE = Fraction(1, 90000)

# MPEG4 part-2 VOP (frame) start code; everything before the first one in the
# first packet is the VOL/VOS header we lift into the stream extradata so the
# MP4 ``esds`` box is well-formed for picky decoders.
_VOP_START = b"\x00\x00\x01\xb6"


class PacketSpool:
    """Append-only, length-prefixed packet store backed by a temp file.

    Bounds memory while collecting a camera's encoded frames during the single
    pass over a (potentially multi-GB) behave file: each packet is written as an
    8-byte big-endian length prefix followed by its bytes, and iterated back in
    order. Iterable yielding ``bytes`` — usable directly by
    :func:`make_camera_video`.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._f = open(path, "wb")
        self.count = 0

    def append(self, data: bytes) -> None:
        self._f.write(struct.pack(">Q", len(data)))
        self._f.write(data)
        self.count += 1

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __iter__(self):
        # Flush any buffered writes so a still-open spool reads back completely
        # (otherwise the last record can be truncated mid-write on disk).
        if self._f is not None:
            self._f.flush()
        with open(self.path, "rb") as f:
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    return
                (n,) = struct.unpack(">Q", hdr)
                yield f.read(n)

    def unlink(self) -> None:
        self.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass


def _extradata(first_packet: bytes) -> bytes:
    """VOL/VOS header bytes preceding the first VOP in the first packet (or b'')."""
    idx = first_packet.find(_VOP_START)
    return first_packet[:idx] if idx > 0 else b""


def make_camera_video(
    packets: Iterable[bytes],
    times_ns: Sequence[int],
    dims: Sequence[Tuple[int, int]],
    out_path,
) -> int:
    """Write ``out_path`` (an MP4) from one camera's encoded frames, 1:1.

    ``packets`` yields the raw MPEG4 frame bytes in record order; ``times_ns[i]``
    is the ``record.time`` of packet ``i``; ``dims[i]`` is its ``(width, height)``.
    All three must be the same length and already filtered to valid frames.

    Returns the number of frames written (== ``len(times_ns)`` on success).
    Tries the lossless PyAV remux first; on any failure falls back to an ffmpeg
    decode+re-encode. Both paths preserve frame↔timestamp correspondence.
    """
    out_path = str(out_path)
    n = len(times_ns)
    if n == 0:
        raise ValueError(f"no frames to write for {out_path}")
    width, height = int(dims[0][0]), int(dims[0][1])

    # `packets` is iterated fresh by each path (a PacketSpool re-opens its file
    # per iteration), so we never materialize a whole camera in RAM.
    try:
        return _remux_pyav(packets, times_ns, width, height, out_path)
    except Exception as exc:  # noqa: BLE001 — fall back on ANY PyAV problem
        print(f"video_remux: PyAV remux failed ({exc!r}); re-encoding with ffmpeg")
        return _reencode_ffmpeg(packets, times_ns, width, height, out_path)


def _remux_pyav(
    packets: Iterable[bytes],
    times_ns: Sequence[int],
    width: int,
    height: int,
    out_path: str,
) -> int:
    """Lossless: mux the original packets with PTS from record.time (VFR)."""
    import av

    tmp = out_path + ".tmp.mp4"
    container = av.open(tmp, mode="w")
    written = 0
    try:
        stream = container.add_stream("mpeg4")
        stream.width = width
        stream.height = height
        stream.time_base = _TIME_BASE
        t0 = int(times_ns[0])
        last_pts = None
        for i, (data, t) in enumerate(zip(packets, times_ns)):
            data = bytes(data)
            if i == 0:
                # MPEG4 VOL/VOS header from the first packet → stream extradata
                # so the MP4 esds box is well-formed for picky decoders.
                extradata = _extradata(data)
                if extradata:
                    stream.codec_context.extradata = extradata
            pkt = av.Packet(data)
            pkt.stream = stream
            pkt.time_base = _TIME_BASE
            pts = round((int(t) - t0) / 1e9 * 90000)
            # record.time is monotonic for valid frames; guard against equal
            # timestamps (would make the muxer reject non-increasing dts).
            if last_pts is not None and pts <= last_pts:
                pts = last_pts + 1
            pkt.pts = pkt.dts = pts
            last_pts = pts
            if i == 0:
                pkt.is_keyframe = True
            container.mux(pkt)
            written += 1
    finally:
        container.close()
    if written != len(times_ns):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise RuntimeError(
            f"packet/timestamp count mismatch for {out_path}: "
            f"muxed {written} vs {len(times_ns)} timestamps"
        )
    os.replace(tmp, out_path)
    return written


def _reencode_ffmpeg(
    packets: Iterable[bytes],
    times_ns: Sequence[int],
    width: int,
    height: int,
    out_path: str,
) -> int:
    """Fallback: decode the contiguous ES and re-encode 1:1 with libx264.

    The packets concatenate into a valid MPEG4 elementary stream (the first
    carries the VOL header), so ffmpeg decodes it directly. ``-vsync 0`` keeps
    one output frame per decoded frame, preserving the 1:1 mapping; the .mat
    timestamps carry the true timing for downstream consumers.
    """
    tmp = out_path + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "m4v", "-i", "pipe:",
        "-vsync", "0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-video_track_timescale", "90000",
        tmp,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for data in packets:
            proc.stdin.write(bytes(data))
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg re-encode failed (rc={rc}) for {out_path}")
    os.replace(tmp, out_path)
    return len(packets)
