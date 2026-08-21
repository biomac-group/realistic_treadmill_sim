#!/usr/bin/env python3
"""Re-filter stored AP-GRF values without rebuilding the contact model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt


def periodic_lowpass(values, sample_rate, cutoff, order=4):
    periodic = np.asarray(values, dtype=float)[:-1]
    b, a = butter(order, cutoff, btype="lowpass", fs=sample_rate)
    tiled = np.tile(periodic, 3)
    filtered = filtfilt(b, a, tiled)[len(periodic) : 2 * len(periodic)]
    return np.r_[filtered, filtered[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-data", type=Path)
    args = parser.parse_args()

    with args.source.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    speeds = sorted({float(row["speed_m_per_s"]) for row in source_rows})
    figure, axes = plt.subplots(len(speeds), 2, figsize=(11, 2.25 * len(speeds)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    output_rows = []
    for row_index, speed in enumerate(speeds):
        for column, side in enumerate(("right", "left")):
            rows = [
                row for row in source_rows
                if float(row["speed_m_per_s"]) == speed and row["side"] == side
            ]
            rows.sort(key=lambda row: float(row["gait_cycle_percent"]))
            phase = np.array([float(row["gait_cycle_percent"]) for row in rows])
            raw = np.array([float(row["ap_grf_raw_n"]) for row in rows])
            sample_rate = float(rows[0]["sample_rate_hz"])
            filtered = periodic_lowpass(raw, sample_rate, args.cutoff, args.order)
            axis = axes[row_index, column]
            axis.plot(phase, raw, color="0.72", linewidth=1, label="Raw")
            axis.plot(phase, filtered, color="#1769aa", linewidth=1.8, label="Low-pass")
            axis.axhline(0, color="0.25", linewidth=0.6)
            axis.grid(alpha=0.2)
            axis.set_title(f"{side.title()}, {speed:.1f} m/s")
            axis.set_ylabel("AP GRF (N)")
            for source_row, filtered_value in zip(rows, filtered):
                saved = dict(source_row)
                saved["ap_grf_lowpass_n"] = filtered_value
                saved["lowpass_cutoff_hz"] = args.cutoff
                saved["lowpass_order"] = args.order
                output_rows.append(saved)
    for axis in axes[-1]:
        axis.set_xlabel("Gait cycle (%)")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        f"P1 ideal treadmill: AP GRF, {args.order}th-order zero-phase Butterworth ({args.cutoff:g} Hz)"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    output_data = args.output_data or args.output.with_suffix(".csv")
    with output_data.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)
    print(output_data)


if __name__ == "__main__":
    main()
