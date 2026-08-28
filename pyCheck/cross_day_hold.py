"""cross_day_hold.py — Hold-stability metrics accumulated per day.

The task's hold requirement is not a constant: `hold_time_s` has ranged from
0.1 s to 0.55 s across days, and it moves together with `target_radius_ratio`
(smaller target + longer hold = harder trial). Any raw "did the hold survive"
rate therefore mixes skill with task demand, so this module tracks both:

* the *demand* actually programmed on each day (required hold, target radius),
* the *task-criterion* stability (fraction of in-target holds that reached
  `hold_complete`, i.e. performance against that day's own requirement),
* a *demand-invariant* stability curve — the survival function of in-target
  dwell time, from which every day can be scored at one common criterion
  (e.g. "P(stay in target ≥ 0.25 s)") no matter what that day required.

Dwell is measured from each `hold_start` (the clock the task itself holds
against; falls back to `target_entry` on recordings without hold_start events)
until the first of `target_exit` / `hold_break` (a broken hold, exact dwell) or
`hold_complete` (dwell = that day's requirement — the hold ended because the
task ended it, so the dwell is right-censored there). Holds still in progress
when the attempt ended are censored at the attempt end.

The survival estimate keeps censoring honest: at criterion *t*, a hold counts
in the denominator only if its fate at *t* is known — every broken hold, plus
censored holds that already survived past *t*. Censored holds shorter than *t*
are excluded rather than counted as failures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Criterion grid for the dwell survival function (seconds). 10 ms resolution up
# to 1 s covers every hold requirement the task has used, and lands exactly on
# the round requirement values (0.10, 0.25, 0.30, 0.45, 0.50).
HOLD_GRID_S: np.ndarray = np.round(np.arange(0.0, 1.0001, 0.01), 4)

# Scalar fields safe for the flat summary CSV (the survival counts are lists and
# live in the JSON only).
HOLD_SUMMARY_FIELDS: List[str] = [
    "n_hold_attempts",
    "n_trials_with_hold",
    "median_required_hold_s",
    "min_required_hold_s",
    "max_required_hold_s",
    "mean_target_radius_ratio",
    "hold_demand_index",
    "task_criterion_hold_success_rate",
    "median_broken_dwell_s",
    "hold_breaks_per_trial",
    "holds_per_trial",
    "mean_excess_acquisition_s",
    "median_excess_acquisition_s",
    "median_hold_phase_fraction",
]

_BREAK_NAMES = ("target_exit", "hold_break")


def _event_time(event: Dict[str, Any]) -> float:
    return float(event.get("time_perf_counter", np.nan))


def _sorted_events(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    usable = [ev for ev in events if np.isfinite(_event_time(ev))]
    return sorted(usable, key=_event_time)


class HoldAccumulator:
    """Collects hold-stability observations across a day's trials."""

    def __init__(self) -> None:
        self._required: List[float] = []
        self._radii: List[float] = []
        self._dwell_s: List[float] = []
        self._censored: List[bool] = []
        self._completed: List[bool] = []
        self._breaks_per_trial: List[float] = []
        self._holds_per_trial: List[float] = []
        self._excess_acquisition_s: List[float] = []
        self._hold_phase_fraction: List[float] = []
        self._n_trials = 0

    def add_trial(
        self,
        attempt: Dict[str, Any],
        events: Sequence[Dict[str, Any]],
        success: bool,
        target_on_s: float = float("nan"),
        hold_complete_s: float = float("nan"),
    ) -> None:
        if not isinstance(attempt, dict):
            return
        required = float(attempt.get("hold_time_s", np.nan))
        radius = float(attempt.get("target_radius_ratio", np.nan))
        if np.isfinite(radius):
            self._radii.append(radius)
        if not np.isfinite(required) or required <= 0:
            return

        ordered = _sorted_events(events)
        clock_name = "hold_start" if any(str(ev.get("name", "")) == "hold_start" for ev in ordered) else "target_entry"
        attempt_end_s = float(attempt.get("end_time_perf_counter", np.nan))
        if not np.isfinite(attempt_end_s) and ordered:
            attempt_end_s = _event_time(ordered[-1])

        n_holds = 0
        n_breaks = 0
        first_clock_s = float("nan")
        for index, event in enumerate(ordered):
            if str(event.get("name", "")) != clock_name:
                continue
            start_s = _event_time(event)
            if not np.isfinite(first_clock_s):
                first_clock_s = start_s
            n_holds += 1
            dwell = float("nan")
            censored = True
            completed = False
            for later in ordered[index + 1:]:
                name = str(later.get("name", ""))
                if name in _BREAK_NAMES:
                    dwell = _event_time(later) - start_s
                    censored = False
                    n_breaks += 1
                    break
                if name == "hold_complete":
                    # Ended by the task at its own criterion: censored at the
                    # requirement, not a failure.
                    dwell = _event_time(later) - start_s
                    completed = True
                    break
            if not np.isfinite(dwell):
                dwell = attempt_end_s - start_s
            if not np.isfinite(dwell) or dwell < 0:
                n_holds -= 1
                continue
            self._required.append(required)
            self._dwell_s.append(float(dwell))
            self._censored.append(bool(censored))
            self._completed.append(bool(completed))

        if n_holds <= 0:
            return

        self._n_trials += 1
        self._holds_per_trial.append(float(n_holds))
        self._breaks_per_trial.append(float(n_breaks))

        if success and np.isfinite(hold_complete_s) and np.isfinite(first_clock_s):
            acquisition_s = hold_complete_s - first_clock_s
            if acquisition_s >= 0:
                # Time spent between the first touch of the target and the
                # completed hold, over and above the hold the task demanded:
                # the price of every broken hold on that trial.
                self._excess_acquisition_s.append(float(acquisition_s - required))
                if np.isfinite(target_on_s) and hold_complete_s > target_on_s:
                    self._hold_phase_fraction.append(
                        float(acquisition_s / (hold_complete_s - target_on_s))
                    )

    def summary(self) -> Dict[str, Any]:
        required = np.asarray(self._required, dtype=float)
        dwell = np.asarray(self._dwell_s, dtype=float)
        censored = np.asarray(self._censored, dtype=bool)
        completed = np.asarray(self._completed, dtype=bool)
        radii = np.asarray(self._radii, dtype=float)

        survived_counts, informative_counts = _survival_counts(dwell, censored, completed, required)
        median_required = float(np.nanmedian(required)) if required.size else float("nan")
        mean_radius = float(np.nanmean(radii)) if radii.size else float("nan")
        broken_dwell = dwell[~censored & ~completed] if dwell.size else dwell

        return {
            "n_hold_attempts": int(dwell.size),
            "n_trials_with_hold": int(self._n_trials),
            "median_required_hold_s": median_required,
            "min_required_hold_s": float(np.nanmin(required)) if required.size else float("nan"),
            "max_required_hold_s": float(np.nanmax(required)) if required.size else float("nan"),
            "mean_target_radius_ratio": mean_radius,
            "hold_demand_index": (
                float(median_required / mean_radius)
                if np.isfinite(median_required) and np.isfinite(mean_radius) and mean_radius > 0
                else float("nan")
            ),
            "task_criterion_hold_success_rate": float(np.mean(completed)) if completed.size else float("nan"),
            "median_broken_dwell_s": float(np.median(broken_dwell)) if broken_dwell.size else float("nan"),
            "hold_breaks_per_trial": float(np.mean(self._breaks_per_trial)) if self._breaks_per_trial else float("nan"),
            "holds_per_trial": float(np.mean(self._holds_per_trial)) if self._holds_per_trial else float("nan"),
            "mean_excess_acquisition_s": (
                float(np.mean(self._excess_acquisition_s)) if self._excess_acquisition_s else float("nan")
            ),
            "median_excess_acquisition_s": (
                float(np.median(self._excess_acquisition_s)) if self._excess_acquisition_s else float("nan")
            ),
            "median_hold_phase_fraction": (
                float(np.median(self._hold_phase_fraction)) if self._hold_phase_fraction else float("nan")
            ),
            "dwell_survived_counts": [int(v) for v in survived_counts],
            "dwell_informative_counts": [int(v) for v in informative_counts],
        }


def _survival_counts(
    dwell: np.ndarray,
    censored: np.ndarray,
    completed: np.ndarray,
    required: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-criterion survivor / informative counts on ``HOLD_GRID_S``.

    A hold is *informative* at criterion t if its fate at t is known: broken
    holds always are; censored holds (task-completed or attempt-ended) only
    while t is within the dwell they are known to have reached.
    """
    survived = np.zeros(HOLD_GRID_S.size, dtype=int)
    informative = np.zeros(HOLD_GRID_S.size, dtype=int)
    if dwell.size == 0:
        return survived, informative

    right_censored = censored | completed
    for grid_index, criterion in enumerate(HOLD_GRID_S):
        reached = dwell >= criterion - 1e-9
        known = reached | ~right_censored
        informative[grid_index] = int(np.count_nonzero(known))
        survived[grid_index] = int(np.count_nonzero(reached & known))
    return survived, informative


def is_estimable_at(row: Dict[str, Any], criterion_s: float) -> bool:
    """Can this day be scored at `criterion_s` without censoring bias?

    Only if the day never demanded a *shorter* hold than the criterion: once the
    task stops a hold at 0.10 s there is no way to know which of those holds
    would have survived to 0.30 s, and the surviving sample is all broken holds
    — biased low. Such days are left blank rather than plotted misleadingly.
    """
    day_min = float(row.get("min_required_hold_s", np.nan))
    return bool(np.isfinite(day_min) and day_min >= float(criterion_s) - 1e-9)


def survival_at(row: Dict[str, Any], criterion_s: float, min_informative: int = 20) -> float:
    """P(in-target dwell ≥ criterion_s) for one day.

    NaN when the day is too thin or the criterion is past its censoring point.
    """
    if not is_estimable_at(row, criterion_s):
        return float("nan")
    survived = np.asarray(row.get("dwell_survived_counts", []), dtype=float)
    informative = np.asarray(row.get("dwell_informative_counts", []), dtype=float)
    if survived.size != HOLD_GRID_S.size or informative.size != HOLD_GRID_S.size:
        return float("nan")
    grid_index = int(np.argmin(np.abs(HOLD_GRID_S - float(criterion_s))))
    if informative[grid_index] < min_informative:
        return float("nan")
    return float(survived[grid_index] / informative[grid_index])


def survival_curve(
    row: Dict[str, Any], max_criterion_s: float, min_informative: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Survival function for one day, truncated where the estimate thins out."""
    survived = np.asarray(row.get("dwell_survived_counts", []), dtype=float)
    informative = np.asarray(row.get("dwell_informative_counts", []), dtype=float)
    if survived.size != HOLD_GRID_S.size or informative.size != HOLD_GRID_S.size:
        return np.empty(0), np.empty(0)
    keep = (HOLD_GRID_S <= max_criterion_s + 1e-9) & (informative >= min_informative)
    if not np.any(keep):
        return np.empty(0), np.empty(0)
    return HOLD_GRID_S[keep], survived[keep] / informative[keep]


def resolve_reference_hold(
    rows: Sequence[Dict[str, Any]],
    override_s: Optional[float] = None,
    coverage: float = 0.8,
) -> float:
    """Common criterion for the demand-invariant comparison.

    A criterion is only estimable on days that never demanded a shorter hold
    (see `is_estimable_at`), so the choice trades coverage against strength: the
    shortest requirement in the selection scores every day but saturates near
    100% once the hold gets easy, and the longest scores almost nothing. Default
    is the criterion that keeps ~`coverage` of days scorable — the toughest
    criterion the bulk of the selection supports.
    """
    if override_s is not None and np.isfinite(float(override_s)) and float(override_s) > 0:
        return float(np.round(float(override_s), 2))
    day_minima = np.asarray(
        [float(row.get("min_required_hold_s", np.nan)) for row in rows], dtype=float
    )
    day_minima = day_minima[np.isfinite(day_minima) & (day_minima > 0)]
    if day_minima.size == 0:
        return 0.25
    criterion = np.quantile(day_minima, np.clip(1.0 - coverage, 0.0, 1.0))
    return float(np.round(np.clip(criterion, HOLD_GRID_S[1], HOLD_GRID_S[-1]), 2))
