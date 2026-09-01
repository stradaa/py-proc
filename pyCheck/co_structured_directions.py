"""co_structured_directions.py — how a structured CO day's targets are grouped.

Both per-day figures key peripheral targets by ``(ring_label, angle)`` rather
than angle alone: a day can run several rings, and near and far reaches to the
same direction are different movements that must not share a panel, a bar or an
average. This module owns that grouping, the ordering, labelling and colouring
that follow from it, and the compass layout the trajectory figure lays its
panels out on.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np

from pyCheck.co_structured import AttemptRecord

# Angles within this many degrees are the same direction. Target angles are
# derived from the config's x_norm/y_norm so they are exact per target; the
# tolerance only absorbs float noise and near-duplicate target placements.
_ANGLE_TOLERANCE_DEG = 5.0

Direction = Tuple[str, float]  # (ring_label, angle_deg)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _cluster_angles(angles: Sequence[float], tolerance: float = _ANGLE_TOLERANCE_DEG) -> List[float]:
    """Cluster centers for a set of angles, in ascending order."""
    ordered = sorted(float(a) for a in angles if np.isfinite(a))
    if not ordered:
        return []
    clusters: List[List[float]] = [[ordered[0]]]
    for angle in ordered[1:]:
        if angle - clusters[-1][-1] <= tolerance:
            clusters[-1].append(angle)
        else:
            clusters.append([angle])
    # 358 deg and 2 deg are the same direction; merge across the wrap point.
    if len(clusters) > 1 and (clusters[0][0] + 360.0) - clusters[-1][-1] <= tolerance:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()
    return [float(np.mean(c) % 360.0) for c in clusters]


def _snap(angle: float, centers: Sequence[float]) -> float:
    """Nearest cluster center to angle, comparing circularly."""
    deltas = [abs((angle - c + 180.0) % 360.0 - 180.0) for c in centers]
    return float(centers[int(np.argmin(deltas))])


def group_by_direction(records: Sequence[AttemptRecord]) -> Dict[Direction, List[AttemptRecord]]:
    """Bucket peripheral records into ``(ring_label, angle)`` directions."""
    centers = _cluster_angles([r.target_angle_deg for r in records])
    groups: Dict[Direction, List[AttemptRecord]] = {}
    for record in records:
        if not np.isfinite(record.target_angle_deg):
            continue
        key = (record.ring_label, _snap(record.target_angle_deg, centers))
        groups.setdefault(key, []).append(record)
    return groups


def direction_order(directions: Sequence[Direction]) -> List[Direction]:
    """Angular order, rings ascending where two rings share an angle."""
    return sorted(directions, key=lambda d: (d[1], d[0]))



def direction_label(direction: Direction, multi_ring: bool) -> str:
    """``"45\u00b0 near"`` on a multi-ring day, plain ``"45\u00b0"`` otherwise."""
    ring, angle = direction
    return f"{angle:.0f}\u00b0 {ring}" if multi_ring and ring else f"{angle:.0f}\u00b0"


def ring_order(labels: Sequence[str]) -> List[str]:
    """Ring labels innermost-first, matching co_structured.ring_labels_sorted."""
    rank = {"ring": 0, "near": 0, "far": 10 ** 6}
    return sorted(set(labels), key=lambda s: rank.get(
        s, int(s[4:]) if s[:4] == "ring" and s[4:].isdigit() else 0))


def direction_colors(directions: Sequence[Direction]) -> Dict[Direction, str]:
    """Cyclic hue by angle, so opposite directions read as opposite colors.

    Rings sharing an angle are separated by lightness, darker for the outer ring.
    """
    rings = ring_order([d[0] for d in directions])
    colors: Dict[Direction, str] = {}
    for ring, angle in directions:
        # Hue carries direction; saturation and value are held fixed so every
        # direction reads with the same visual weight (a raw hsv colormap makes
        # yellow far lighter than blue). Rings differ only in value.
        value = 0.80
        if len(rings) > 1:
            frac = rings.index(ring) / (len(rings) - 1)
            value = 0.92 - 0.30 * frac  # inner ring lighter, outer darker
        rgb = matplotlib.colors.hsv_to_rgb([(angle % 360.0) / 360.0, 0.85, value])
        colors[(ring, angle)] = matplotlib.colors.to_hex(rgb)
    return colors


# ---------------------------------------------------------------------------
# Compass layout
# ---------------------------------------------------------------------------

def _grid_size(n_directions: int) -> int:
    """Smallest odd grid whose perimeter can hold every direction."""
    size = 3
    while 4 * (size - 1) < n_directions:
        size += 2
    return size


def _perimeter_cells(size: int) -> List[Tuple[Tuple[int, int], float]]:
    """Perimeter ``((row, col), angle_deg)`` of a size x size grid, y-up."""
    center = (size - 1) / 2.0
    cells = []
    for row in range(size):
        for col in range(size):
            if 0 < row < size - 1 and 0 < col < size - 1:
                continue
            angle = float(np.degrees(np.arctan2(center - row, col - center)) % 360.0)
            cells.append(((row, col), angle))
    return cells


def compass_cells(directions: Sequence[Direction]) -> Tuple[int, Dict[Direction, Tuple[int, int]]]:
    """Map each direction onto the perimeter cell pointing the same way.

    Solved as an assignment problem over circular angular distance so the
    mapping is an optimal bijection rather than a greedy walk — with 16
    directions on a 5x5 grid this reproduces the exact compass (0 deg to
    mid-right, 45 deg to the corner, 90 deg to mid-top, and the 22.5 deg
    offsets to the cells between them).
    """
    from scipy.optimize import linear_sum_assignment

    ordered = direction_order(directions)
    size = _grid_size(len(ordered))
    cells = _perimeter_cells(size)
    cost = np.array([
        [abs((angle - cell_angle + 180.0) % 360.0 - 180.0) for _, cell_angle in cells]
        for _, angle in ordered
    ])
    rows, cols = linear_sum_assignment(cost)
    return size, {ordered[r]: cells[c][0] for r, c in zip(rows, cols)}
