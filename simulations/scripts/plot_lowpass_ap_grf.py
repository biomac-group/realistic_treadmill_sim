#!/usr/bin/env python3
"""Plot raw and low-pass-filtered AP GRFs from ideal-treadmill solutions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_walking2d import (
    extract_solution,
    load_model_with_treadmill_contact,
    load_solution,
    reconstruct_accelerations,
)


def periodic_lowpass(values: np.ndarray, sample_rate: float, cutoff: float, order: int):
    """Zero-phase Butterworth filtering with periodic gait-cycle padding."""
    values = np.asarray(values, dtype=float)
    periodic = values[:-1]
    if cutoff >= sample_rate / 2:
        raise ValueError(f"Cutoff {cutoff:g} Hz must be below Nyquist ({sample_rate / 2:g} Hz).")
    sos = butter(order, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    tiled = np.tile(periodic, 3)
    filtered = sosfiltfilt(sos, tiled)[len(periodic) : 2 * len(periodic)]
    return np.r_[filtered, filtered[0]]


def ap_grfs(result_file: Path):
    states, globals_, _, settings = extract_solution(load_solution(result_file))
    duration = float(np.asarray(globals_.dur))
    states = reconstruct_accelerations(states, duration)
    model_file = settings["settings"]["model"]
    biosym_model = load_model_with_treadmill_contact(model_file=model_file)
    contact = biosym_model.gc_model

    def evaluate(state):
        full_state = state.replace(
            tau=jnp.zeros(biosym_model.tau.n),
            ext_forces=jnp.zeros(biosym_model.ext_forces.n),
            ext_torques=jnp.zeros(biosym_model.ext_torques.n),
        )
        return contact._contact_positions_and_forces(
            full_state, biosym_model.default_constants
        )[1]

    sphere_forces = np.asarray(jax.vmap(evaluate)(states))
    belt_slots = np.asarray(contact._belt_slots)
    right = sphere_forces[:, belt_slots == 0, 0].sum(axis=1)
    left = sphere_forces[:, belt_slots == 1, 0].sum(axis=1)
    return duration, right, left


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--speeds", nargs="+", type=float, default=[1.0, 1.2, 1.4, 1.6, 1.8])
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-data",
        type=Path,
        help="CSV output (default: the plot path with a .csv suffix)",
    )
    args = parser.parse_args()

    files = {
        speed: args.input_dir / f"treadmill_ideal_{speed:.1f}".replace(".", "_")
        for speed in args.speeds
    }
    files = {speed: path.with_suffix(".pkl") for speed, path in files.items()}
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ideal-treadmill results:\n  " + "\n  ".join(missing))

    figure, axes = plt.subplots(
        len(files), 2, figsize=(11, 2.25 * len(files)), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    phase = None
    rows = []
    for row, (speed, result_file) in enumerate(files.items()):
        duration, right, left = ap_grfs(result_file)
        sample_rate = (len(right) - 1) / duration
        phase = np.linspace(0, 100, len(right))
        for column, (side, raw) in enumerate((("Right", right), ("Left", left))):
            filtered = periodic_lowpass(raw, sample_rate, args.cutoff, args.order)
            rows.extend(
                {
                    "speed_m_per_s": speed,
                    "side": side.lower(),
                    "gait_cycle_percent": gait_cycle,
                    "duration_s": duration,
                    "sample_rate_hz": sample_rate,
                    "ap_grf_raw_n": raw_value,
                    "ap_grf_lowpass_n": filtered_value,
                    "lowpass_cutoff_hz": args.cutoff,
                    "lowpass_order": args.order,
                }
                for gait_cycle, raw_value, filtered_value in zip(phase, raw, filtered)
            )
            axis = axes[row, column]
            axis.plot(phase, raw, color="0.72", linewidth=1.0, label="Raw")
            axis.plot(phase, filtered, color="#1769aa", linewidth=1.8, label="Low-pass")
            axis.axhline(0, color="0.25", linewidth=0.6)
            axis.set_title(f"{side}, {speed:.1f} m/s")
            axis.set_ylabel("AP GRF (N)")
            axis.grid(alpha=0.2)

    for axis in axes[-1]:
        axis.set_xlabel("Gait cycle (%)")
    axes[0, 0].legend(frameon=False, loc="best")
    figure.suptitle(
        f"P1 ideal treadmill: AP GRF, {args.order}th-order zero-phase Butterworth "
        f"({args.cutoff:g} Hz)"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    data_path = args.output_data or args.output.with_suffix(".csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    print(data_path)


if __name__ == "__main__":
    main()
