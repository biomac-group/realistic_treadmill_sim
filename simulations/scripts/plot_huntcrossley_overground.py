"""Plot angles, net joint moments, and GRFs from saved Hunt-Crossley results."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np

from biosym.constraints.dynamics import calc_forces
from biosym.model.model import load_model
from biosym.utils import read_mot

from src.contact_huntcrossley import install as install_huntcrossley_contact
from visualize_walking2d import (
    create_slip_friction_figure,
    slip_friction_output_path,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Saved overground pickle(s).")
    parser.add_argument("--participant", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Replace existing plots.")
    return parser.parse_args()


def objective_file(settings, name):
    for objective in settings.get("objectives", []):
        if objective.get("name") == name:
            return objective.get("args", {}).get("presegmented_file")
    return None


def result_speed(path):
    match = re.search(r"overground_(\d+)_(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot infer speed from {path.name}")
    return float(f"{match.group(1)}.{match.group(2)}")


def resample(values, n_nodes):
    values = np.asarray(values, dtype=float)
    if len(values) == n_nodes:
        return values
    old = np.linspace(0.0, 1.0, len(values), endpoint=False)
    new = np.linspace(0.0, 1.0, n_nodes, endpoint=False)
    return np.interp(new, np.append(old, 1.0), np.append(values, values[0]))


def plot_result(path, participant, force=False):
    install_huntcrossley_contact()
    angle_output = path.with_name(f"{path.stem}_joint_angles_moments.png")
    grf_output = path.with_name(f"{path.stem}_grf.png")
    slip_output = Path(slip_friction_output_path(grf_output))
    if not force and angle_output.exists() and grf_output.exists() and slip_output.exists():
        print(f"Keeping existing plots for {path}")
        return

    with path.open("rb") as handle:
        (states, globals_dict), info, settings = cloudpickle.load(handle)
    model_file = settings["settings"]["model"]
    model = load_model(model_file, force_rebuild=True)
    n_nodes = len(states)
    phase = np.linspace(0.0, 100.0, n_nodes)
    speed = result_speed(path)

    angle_file = objective_file(settings, "track_angles")
    if angle_file is None:
        cadence = "natural" if speed <= 1.4 else "fast"
        angle_file = (
            "data/opensim_exports/winter_1987/"
            f"winter_1987_{cadence}_cadence_kinematics_mean_var_rad.mot"
        )
    angle_data = read_mot(angle_file)
    q = np.asarray(states.q, dtype=float)
    moments = np.asarray(
        model.run["actuator_model"](states, model.default_constants), dtype=float
    )
    rows = (
        (("q_hip_r", "hip_flexion_r_mean", 0, "Right hip"), ("q_hip_l", "hip_flexion_l_mean", 3, "Left hip")),
        (("q_knee_r", "knee_angle_r_mean", 1, "Right knee"), ("q_knee_l", "knee_angle_l_mean", 4, "Left knee")),
        (("q_ankle_r", "ankle_angle_r_mean", 2, "Right ankle"), ("q_ankle_l", "ankle_angle_l_mean", 5, "Left ankle")),
    )
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True)
    for row, sides in enumerate(rows):
        for side, (q_name, data_name, moment_index, title) in enumerate(sides):
            angle_axis, moment_axis = axes[row, side * 2 : side * 2 + 2]
            experimental = np.rad2deg(resample(angle_data[data_name], n_nodes))
            variance_name = data_name.replace("_mean", "_var")
            experimental_std = np.rad2deg(
                np.sqrt(np.maximum(resample(angle_data[variance_name], n_nodes), 0.0))
            )
            angle_axis.plot(phase, np.rad2deg(q[:, model.coordinates.names.index(q_name)]), label="Optimized", linewidth=2)
            angle_axis.plot(phase, experimental, ":", label="Experimental mean", linewidth=2)
            angle_axis.fill_between(
                phase,
                experimental - experimental_std,
                experimental + experimental_std,
                alpha=0.2,
                label="Experimental ±1 SD",
            )
            angle_axis.set_title(f"{title} angle")
            angle_axis.set_ylabel("Angle (deg)")
            angle_axis.grid(True, alpha=0.3)
            moment_axis.plot(phase, moments[:, moment_index], linewidth=2)
            moment_axis.axhline(0.0, color="k", linewidth=0.7)
            moment_axis.set_title(f"{title} net moment")
            moment_axis.set_ylabel("Moment (N m)")
            moment_axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Gait cycle (%)")
    axes[0, 0].legend()
    fig.suptitle(f"P{participant}, {speed:.1f} m/s")
    fig.tight_layout()
    fig.savefig(angle_output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    evaluated = []
    for node in range(n_nodes):
        evaluated.append(np.asarray(calc_forces(states[node], model).ext_forces).reshape(2, 3))
    evaluated = np.asarray(evaluated)
    token = f"{speed:.1f}".replace(".", "p")
    grf_data = read_mot(f"data/opensim_exports/p{participant}_grf_mean_speed_{token}.mot")
    curves = (
        (0, 0, "grf_x_r_mean", "Right horizontal GRF"),
        (0, 1, "grf_y_r_mean", "Right vertical GRF"),
        (1, 0, "grf_x_l_mean", "Left horizontal GRF"),
        (1, 1, "grf_y_l_mean", "Left vertical GRF"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (foot, component, name, title) in zip(axes.flat, curves):
        axis.plot(phase, evaluated[:, foot, component], label="Optimized", linewidth=2)
        axis.plot(phase, resample(grf_data[name], n_nodes), ":", label="Experimental", linewidth=2)
        axis.set_title(title)
        axis.set_ylabel("Force (N)")
        axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Gait cycle (%)")
    axes[0, 0].legend()
    fig.suptitle(f"P{participant}, {speed:.1f} m/s")
    fig.tight_layout()
    fig.savefig(grf_output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    slip_figure = create_slip_friction_figure(model, states)
    slip_figure.savefig(slip_output, dpi=180, bbox_inches="tight")
    plt.close(slip_figure)
    print(f"Saved {angle_output}")
    print(f"Saved {grf_output}")
    print(f"Saved {slip_output}")


def main():
    args = parse_args()
    for result in args.results:
        plot_result(Path(result), args.participant, args.force)


if __name__ == "__main__":
    main()
