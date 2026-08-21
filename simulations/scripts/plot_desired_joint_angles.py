import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_mot_table(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]

    endheader_index = lines.index("endheader")
    columns = lines[endheader_index + 1].split()
    data = np.vstack([np.fromstring(line, sep="\t") for line in lines[endheader_index + 2 :]])
    if data.shape[1] != len(columns):
        data = np.vstack([np.fromstring(line, sep=" ") for line in lines[endheader_index + 2 :]])
    return columns, data


def find_track_angles_config(tracking_config):
    with open(tracking_config, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    collocation = payload.get("collocation", payload)
    for objective in collocation.get("objectives", []):
        if objective.get("name") != "track_angles":
            continue
        args = objective.get("args", {})
        candidate = args.get("presegmented_file")
        exclude = args.get("exclude", [])
        if candidate:
            return candidate, exclude
    return None, []


def collect_joint_series(columns, data, exclude):
    column_lookup = {name: index for index, name in enumerate(columns)}
    time = data[:, column_lookup["time"]]

    joint_names = []
    for column in columns:
        if not column.endswith("_mean"):
            continue
        base = column[:-5]
        if base == "time" or base in exclude:
            continue
        var_column = f"{base}_var"
        if var_column not in column_lookup:
            continue
        joint_names.append(base)

    series = []
    for name in joint_names:
        mean_values = data[:, column_lookup[f"{name}_mean"]]
        var_values = data[:, column_lookup[f"{name}_var"]]
        std_values = np.sqrt(np.maximum(var_values, 0.0))
        series.append((name, mean_values, std_values))
    return time, series


def create_joint_angle_figure(time, series, show_variance):
    n_plots = len(series)
    ncols = 2 if n_plots > 1 else 1
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)

    for axis, (name, mean_values, std_values) in zip(axes, series):
        axis.plot(time, mean_values, linewidth=2, label=f"{name} mean")
        if show_variance:
            axis.fill_between(
                time,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.25,
                label=f"{name} ±1 std",
            )
        axis.set_title(name)
        axis.set_ylabel("Angle")
        axis.grid(True, alpha=0.3)
        axis.legend()

    for axis in axes[n_plots:]:
        axis.axis("off")

    for axis in axes[-ncols:]:
        if axis.has_data():
            axis.set_xlabel("Time (s)")

    fig.suptitle("Desired Joint Angles")
    fig.tight_layout()
    return fig


def save_figure(fig, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot desired joint angles from a presegmented MOT file.")
    parser.add_argument(
        "--mot-file",
        default=None,
        help="Path to the desired kinematics MOT file. If omitted, use --tracking-config.",
    )
    parser.add_argument(
        "--tracking-config",
        default=None,
        help="Tracking YAML file from which to read the track_angles presegmented_file.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path for saving the figure.",
    )
    parser.add_argument(
        "--no-variance",
        action="store_true",
        help="Plot only the mean trajectories without the variance band.",
    )
    args = parser.parse_args()

    mot_file = args.mot_file
    exclude = []
    if mot_file is None:
        if args.tracking_config is None:
            raise ValueError("Provide either --mot-file or --tracking-config.")
        mot_file, exclude = find_track_angles_config(args.tracking_config)
        if mot_file is None:
            raise ValueError("Could not find a track_angles presegmented_file in the tracking config.")

    columns, data = load_mot_table(mot_file)
    time, series = collect_joint_series(columns, data, exclude)
    if not series:
        raise ValueError(f"No joint-angle mean/var columns found in {mot_file}.")

    fig = create_joint_angle_figure(time, series, show_variance=not args.no_variance)
    print(f"Desired joint-angle source: {mot_file}")
    print(f"Plotted joints: {', '.join(name for name, _, _ in series)}")

    if args.save:
        save_figure(fig, args.save)
        print(f"Saved figure to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
