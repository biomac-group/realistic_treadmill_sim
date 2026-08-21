"""Create separate GRF and belt-speed figures with a raw-AP controller replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.gait_data import DEFAULT_GAIT_DATA_FILE
from src.treadmill_controller import cubic_bridge_post_hs, controller_next_speed
from visualize_walking2d import (
    get_treadmill_controller_metadata,
    load_desired_belt_speed,
    load_desired_grf,
)


def load_contact_grfs(path: Path, n_nodes: int) -> dict[str, np.ndarray]:
    """Sum contact-point forces into right and left resultant GRFs."""
    contacts = pd.read_csv(path)
    expected_nodes = np.arange(n_nodes)
    grfs: dict[str, np.ndarray] = {}
    for side, suffix in (("right", "r"), ("left", "l")):
        side_rows = contacts.loc[contacts["side"] == side]
        summed = side_rows.groupby("node").agg(
            fx=("friction_force_x_N", "sum"),
            fy=("normal_force_N", "sum"),
        ).reindex(expected_nodes)
        if summed.isna().any().any():
            raise ValueError(f"Missing {side} contact-force nodes in {path}.")
        grfs[f"fx_{suffix}"] = summed["fx"].to_numpy()
        grfs[f"fy_{suffix}"] = summed["fy"].to_numpy()
    return grfs


def periodic_with_endpoint(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.r_[values, values[0]]


def interpolated_ap_grf(
    values: np.ndarray,
    heel_strike_index: int,
    interpolation_samples: int,
) -> np.ndarray:
    periodic = np.asarray(values, dtype=float)[:-1]
    interpolated = np.asarray(
        cubic_bridge_post_hs(
            periodic, heel_strike_index, interpolation_samples
        )
    )
    return periodic_with_endpoint(interpolated)


def replay_without_interpolation(
    *,
    fy: np.ndarray,
    fx: np.ndarray,
    speed: np.ndarray,
    metadata: dict,
) -> np.ndarray:
    """Evaluate one controller update at every node using the raw AP GRF."""
    fy_periodic = np.asarray(fy, dtype=float)[:-1]
    fx_periodic = np.asarray(fx, dtype=float)[:-1]
    speed_periodic = np.asarray(speed, dtype=float)[:-1]
    n_periodic = speed_periodic.size
    predicted_next = np.asarray(
        controller_next_speed(
            fy=fy_periodic,
            fx=fx_periodic,
            speed=speed_periodic,
            target_speed=float(metadata["target_speed"]),
            dt=float(metadata["dt"]),
            delay_samples=int(metadata["delay"]),
            error_delay_samples=int(metadata["error_delay_samples"]),
            k_fy=float(metadata["k_fy"]),
            k_fx=float(metadata["k_fx"]),
            k_p=float(metadata["k_p"]),
            k_d=float(metadata["k_d"]),
            swing_force_threshold=float(metadata["swing_force_threshold"]),
            post_hs_interpolation_samples=0,
        )
    )
    aligned = np.empty_like(predicted_next)
    aligned[(np.arange(n_periodic) + 1) % n_periodic] = predicted_next
    return periodic_with_endpoint(aligned)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--desired-grf", type=Path, required=True)
    parser.add_argument("--participant", type=int, required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    with args.result.open("rb") as handle:
        ((states, globals_), _info, settings) = cloudpickle.load(handle)

    treadmill_states = np.asarray(states.gc_model, dtype=float)
    n_nodes = treadmill_states.shape[0]
    nodes = np.arange(n_nodes)
    grf = load_contact_grfs(args.contacts, n_nodes)
    desired_grf = load_desired_grf(str(args.desired_grf), n_nodes)
    desired_belt = load_desired_belt_speed(
        str(DEFAULT_GAIT_DATA_FILE), args.participant, args.speed, n_nodes
    )
    metadata = get_treadmill_controller_metadata(settings).copy()
    metadata["dt"] = float(globals_.dur) / (n_nodes - 1)

    default_samples = int(metadata.get("post_hs_interpolation_samples", 0))
    side_metadata = {
        "r": {
            "heel_strike_index": int(metadata.get("heel_strike_index_r", 0)),
            "interpolation_samples": int(
                metadata.get("post_hs_interpolation_samples_r", default_samples)
            ),
        },
        "l": {
            "heel_strike_index": int(
                metadata.get("heel_strike_index_l", (n_nodes - 1) // 2)
            ),
            "interpolation_samples": int(
                metadata.get("post_hs_interpolation_samples_l", default_samples)
            ),
        },
    }

    output_prefix = args.output_prefix or args.result.with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Keep the original component colors. Use purple only over the AP samples
    # that the controller actually replaces by interpolation.
    grf_colors = {"fx_r": "C0", "fy_r": "C1", "fx_l": "C2", "fy_l": "C3"}
    interpolation_color = "C4"
    fig_grf, ax_grf = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    grf_labels = {
        "fx_r": "Predicted right AP GRF",
        "fy_r": "Predicted right vertical GRF",
        "fx_l": "Predicted left AP GRF",
        "fy_l": "Predicted left vertical GRF",
    }
    desired_keys = {
        "fx_r": "grf_x_r", "fy_r": "grf_y_r",
        "fx_l": "grf_x_l", "fy_l": "grf_y_l",
    }
    for key in ("fx_r", "fy_r", "fx_l", "fy_l"):
        ax_grf.plot(nodes, grf[key], color=grf_colors[key], label=grf_labels[key])
        ax_grf.plot(
            nodes,
            desired_grf[desired_keys[key]],
            color=grf_colors[key],
            linestyle=":",
            linewidth=2,
            label=grf_labels[key].replace("Predicted", "Experimental"),
        )
    for side, side_name, linestyle in (("r", "right", "--"), ("l", "left", "-.")):
        side_info = side_metadata[side]
        interpolated = interpolated_ap_grf(
            grf[f"fx_{side}"],
            side_info["heel_strike_index"],
            side_info["interpolation_samples"],
        )
        interpolation_nodes = (
            side_info["heel_strike_index"]
            + np.arange(side_info["interpolation_samples"])
        ) % (n_nodes - 1)
        interpolation_only = np.full(n_nodes, np.nan)
        interpolation_only[interpolation_nodes] = interpolated[interpolation_nodes]
        # Repeat node zero at the periodic endpoint when it was interpolated.
        if 0 in interpolation_nodes:
            interpolation_only[-1] = interpolated[-1]
        ax_grf.plot(
            nodes,
            interpolation_only,
            color=interpolation_color,
            linestyle=linestyle,
            linewidth=2.5,
            label=f"Interpolated {side_name} AP GRF",
        )
    ax_grf.set(xlabel="Node", ylabel="Force (N)")
    ax_grf.grid(True, alpha=0.3)
    ax_grf.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=3, fontsize=9, frameon=False,
    )
    grf_base = Path(f"{output_prefix}_grf")
    save_figure(fig_grf, grf_base)
    plt.close(fig_grf)

    replay_r = replay_without_interpolation(
        fy=grf["fy_r"], fx=grf["fx_r"], speed=treadmill_states[:, 0],
        metadata=metadata,
    )
    replay_l = replay_without_interpolation(
        fy=grf["fy_l"], fx=grf["fx_l"], speed=treadmill_states[:, 1],
        metadata=metadata,
    )

    fig_belt, ax_belt = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    belt_colors = {"r": "C0", "l": "C1"}
    for column, side, side_name, replay in (
        (0, "r", "Right", replay_r),
        (1, "l", "Left", replay_l),
    ):
        color = belt_colors[side]
        ax_belt.plot(
            nodes, -treadmill_states[:, column], color=color,
            label=f"{side_name} belt speed (interpolated controller)",
        )
        ax_belt.plot(
            nodes, desired_belt[side_name.lower()], color=color,
            linestyle=":", linewidth=2, label=f"Desired {side_name.lower()} belt speed",
        )
        ax_belt.plot(
            nodes, -replay, color=color, linestyle="--", linewidth=2,
            label=f"{side_name} belt speed (raw-AP one-pass replay)",
        )
    ax_belt.set(xlabel="Node", ylabel="Belt speed (m/s)")
    ax_belt.grid(True, alpha=0.3)
    ax_belt.legend(ncol=2, fontsize=9)
    belt_base = Path(f"{output_prefix}_belt_speed_comparison")
    save_figure(fig_belt, belt_base)
    plt.close(fig_belt)

    print(f"Saved {grf_base.with_suffix('.png')}")
    print(f"Saved {grf_base.with_suffix('.pdf')}")
    print(f"Saved {belt_base.with_suffix('.png')}")
    print(f"Saved {belt_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
