#!/usr/bin/env python3
"""Compare post-heel-strike AP interpolation with optional low-pass filtering."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.treadmill_controller import periodic_butterworth_filtfilt


def load_forces(path):
    forces = defaultdict(lambda: np.zeros((101, 2)))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            node = int(row["node"])
            side = row["side"]
            forces[side][node, 0] += float(row["friction_force_x_N"])
            forces[side][node, 1] += float(row["normal_force_N"])
    return forces


def principal_hs(fy, threshold=10.0):
    periodic = fy[:-1]
    contact = periodic >= threshold
    candidates = np.flatnonzero(contact & ~np.roll(contact, 1))
    scores = [sum(periodic[(node + np.arange(15)) % len(periodic)]) for node in candidates]
    return int(candidates[int(np.argmax(scores))])


def interpolate_post_hs(fx, hs, first_offset=2, last_offset=5):
    periodic = np.asarray(fx[:-1], dtype=float)
    n = len(periodic)
    offsets = np.arange(-3, 11)
    values = periodic[(hs + offsets) % n]
    missing = (offsets >= first_offset) & (offsets <= last_offset)
    interpolator = PchipInterpolator(offsets[~missing], values[~missing])
    result = periodic.copy()
    result[(hs + offsets[missing]) % n] = interpolator(offsets[missing])
    return np.r_[result, result[0]], (hs + offsets[missing]) % n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    forces = load_forces(args.diagnostics)
    dt = args.duration / 100.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    saved_rows = []
    for row_index, side in enumerate(("right", "left")):
        fx, fy = forces[side].T
        hs = principal_hs(fy)
        interpolated, replaced = interpolate_post_hs(fx, hs)
        filtered = np.asarray(periodic_butterworth_filtfilt(interpolated[:-1], dt, args.cutoff))
        filtered = np.r_[filtered, filtered[0]]
        phase = np.arange(len(fx))
        for axis in axes[row_index]:
            axis.plot(phase, fx, color="0.65", linewidth=1.2, label="Raw")
            axis.plot(phase, interpolated, color="#d67c00", linewidth=2, label="PCHIP only")
            axis.plot(phase, filtered, color="#1769aa", linewidth=2, label=f"PCHIP + {args.cutoff:g} Hz")
            axis.scatter(replaced, interpolated[replaced], color="#d67c00", zorder=4, s=25)
            axis.axhline(0, color="0.25", linewidth=0.6)
            axis.grid(alpha=0.2)
        axes[row_index, 0].set_xlim(0, 100)
        axes[row_index, 0].set_title(f"{side.title()}: full cycle")
        axes[row_index, 1].set_xlim(max(0, hs - 3), min(100, hs + 12))
        axes[row_index, 1].set_title(f"{side.title()}: post-HS detail (HS node {hs})")
        axes[row_index, 0].set_ylabel("AP GRF (N)")
        for node in range(len(fx)):
            saved_rows.append({"side": side, "node": node, "raw_ap_grf_n": fx[node], "pchip_ap_grf_n": interpolated[node], "pchip_lowpass_ap_grf_n": filtered[node], "interpolated_node": int(node in replaced)})
    for axis in axes[-1]:
        axis.set_xlabel("Node")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Post-heel-strike AP artifact: interpolation simulation")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(saved_rows[0]))
        writer.writeheader(); writer.writerows(saved_rows)
    print(args.output)


if __name__ == "__main__":
    main()
