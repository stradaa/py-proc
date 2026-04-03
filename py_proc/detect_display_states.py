"""
detect_display_states.py — Port of m260216_detect_display_states.m.

Reads recNNN.display.dat, thresholds, identifies real high/low states by
minimum-duration filtering, saves recNNN.display_corrected.dat, and writes
a diagnostic PNG.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

from .helpers import rec_paths, get_recs


def detect_display_states(day, monkeydir, out_suffix=''):
    """
    Port of detect_display_states(day).

    Configuration mirrors m260216_detect_display_states.m:
      fs = 30000 Hz
      threshold = 0.2 V
      min_high_duration_ms = 10 ms
      min_low_duration_ms = 10 ms

    Output: recNNN.display_corrected.dat (float32, 0 or 1)
            detect_display_states_recNNN.png in claude/figures/
    """
    day_dir = os.path.join(monkeydir, day)
    recs = get_recs(day_dir)

    fs = 30000
    threshold = 0.2          # volts
    min_high_ms = 10.0        # ms
    min_low_ms = 10.0         # ms

    fig_dir = os.path.join(monkeydir, 'claude', 'figures', day)
    os.makedirs(fig_dir, exist_ok=True)

    for rec in recs:
        _, rec_dir, rec_pref, _ = rec_paths(day, rec, monkeydir)
        py_pref = rec_pref + out_suffix
        dat_file = f'{rec_pref}.display.dat'

        if not os.path.exists(dat_file):
            print(f'  WARNING: {dat_file} not found, skipping')
            continue

        print(f'\n{"="*64}')
        print(f'rec{rec}')
        print(f'{"="*64}')

        raw = np.fromfile(dat_file, dtype=np.float32)
        n_samples = len(raw)
        duration_s = n_samples / fs
        print(f'  {n_samples} samples, {duration_s:.1f} s')

        # --- Threshold ---
        above = raw > threshold

        # --- Find contiguous above-threshold episodes ---
        d = np.diff(above.astype(np.int8))
        rising = np.where(d == 1)[0] + 1
        falling = np.where(d == -1)[0]

        if above[0]:
            rising = np.concatenate([[0], rising])
        if above[-1]:
            falling = np.concatenate([falling, [n_samples - 1]])

        n_ep = min(len(rising), len(falling))
        rising = rising[:n_ep]
        falling = falling[:n_ep]
        episode_dur_ms = (falling - rising) / fs * 1000.0

        print(f'  Total above-threshold episodes: {n_ep}')

        # --- Accept only long episodes ---
        is_real_high = episode_dur_ms >= min_high_ms
        high_starts = rising[is_real_high]
        high_ends = falling[is_real_high]
        high_dur_ms = episode_dur_ms[is_real_high]

        print(f'  Real high states (>={min_high_ms} ms): {len(high_starts)}')
        if len(high_dur_ms) > 0:
            print(f'    Mean duration: {np.mean(high_dur_ms):.1f} ms')
            print(f'    Median duration: {np.median(high_dur_ms):.1f} ms')
            print(f'    Min duration: {np.min(high_dur_ms):.1f} ms')
            print(f'    Max duration: {np.max(high_dur_ms):.1f} ms')

        # --- Low states (gaps between high states) ---
        if len(high_starts) > 1:
            low_starts = high_ends[:-1]
            low_ends = high_starts[1:]
            low_dur_ms = (low_ends - low_starts) / fs * 1000.0
            is_real_low = low_dur_ms >= min_low_ms
            low_starts = low_starts[is_real_low]
            low_ends = low_ends[is_real_low]
            low_dur_ms = low_dur_ms[is_real_low]
            print(f'  Real low states (>={min_low_ms} ms): {len(low_starts)}')

        # --- Build corrected signal ---
        display_corrected = np.zeros(n_samples, dtype=np.float32)
        for hs, he in zip(high_starts, high_ends):
            display_corrected[hs:he + 1] = 1.0

        # --- Save ---
        out_file = f'{py_pref}.display_corrected.dat'
        display_corrected.tofile(out_file)
        print(f'  Saved {out_file}')

        # --- Diagnostic plot ---
        _plot_display_states(
            raw, display_corrected, high_starts, high_ends,
            episode_dur_ms, rec, day, fs, threshold, min_high_ms,
            py_pref, fig_dir)

    print('\ndetect_display_states: Done.')


def _plot_display_states(raw, corrected, high_starts, high_ends,
                          episode_dur_ms, rec, day, fs, threshold, min_high_ms,
                          rec_pref, fig_dir):
    """Save a diagnostic 3-panel PNG."""
    n_samples = len(raw)
    t = np.arange(n_samples) / fs

    # Find a 20-second window to plot
    events_file = f'{rec_pref}.Events.mat'
    if os.path.exists(events_file):
        try:
            ev_data = loadmat(events_file, simplify_cells=True)
            Events = ev_data.get('Events', {})
            success_idx = np.where(
                (np.asarray(Events.get('Success', [])) == 1) &
                ~np.isnan(np.asarray(Events.get('StartAq', [np.nan])))
            )[0]
            if len(success_idx) > 0:
                mid = success_idx[len(success_idx) // 2]
                trial_ms = float(Events['StartAq'][mid])
                trial_samp = round(trial_ms / 1000 * fs)
                plot_start = max(0, trial_samp - 10 * fs)
            else:
                plot_start = max(0, int(high_starts[0])) if len(high_starts) > 0 else 0
        except Exception:
            plot_start = max(0, int(high_starts[0])) if len(high_starts) > 0 else 0
    else:
        plot_start = max(0, int(high_starts[0])) if len(high_starts) > 0 else 0

    plot_end = min(plot_start + 20 * fs, n_samples)
    idx = np.arange(plot_start, plot_end)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # Panel 1: raw + threshold
    axes[0].plot(t[idx], raw[idx], linewidth=0.3, color='#456db4')
    axes[0].axhline(threshold, color='r', linestyle='--', linewidth=1.5)
    axes[0].set_title(f'rec{rec} — Raw signal (20 s around trial) + {threshold:.1f} V threshold')
    axes[0].set_ylabel('V')
    axes[0].set_xlabel('Time (s)')
    axes[0].grid(True)

    # Panel 2: detected high states
    axes[1].plot(t[idx], raw[idx], linewidth=0.3, color='#aaaaaa')
    for hs, he in zip(high_starts, high_ends):
        if hs > plot_end or he < plot_start:
            continue
        hs2 = max(int(hs), plot_start)
        he2 = min(int(he), plot_end - 1)
        axes[1].axvspan(t[hs2], t[he2], alpha=0.3, color='#45b44a', linewidth=0)
    axes[1].plot(t[idx], raw[idx], linewidth=0.3, color='#456db4')
    axes[1].set_title(f'Detected high states (green, min {min_high_ms} ms) — {len(high_starts)} total')
    axes[1].set_ylabel('V')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True)

    # Panel 3: episode duration histogram
    if len(episode_dur_ms) > 0:
        axes[2].hist(episode_dur_ms, bins=200, color='#456db4', edgecolor='none')
        axes[2].axvline(min_high_ms, color='r', linestyle='--', linewidth=2)
        axes[2].set_xscale('log')
        axes[2].set_title('Above-threshold episode durations (log scale)')
        axes[2].set_xlabel('Duration (ms)')
        axes[2].set_ylabel('Count')
        axes[2].grid(True)

    fig.suptitle(f'Display State Detection — rec{rec} | day {day} | '
                 f'thresh={threshold:.1f} V | min_high={min_high_ms} ms',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    png_path = os.path.join(fig_dir, f'detect_display_states_rec{rec}.png')
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    print(f'  Saved diagnostic plot: {png_path}')
