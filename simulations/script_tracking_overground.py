import jax
jax.config.update('jax_enable_x64', True)
import os
from pathlib import Path

from biosym.ocp.collocation import Collocation
from biosym.model import model
from biosym.constraints.dynamics import calc_forces
import xml.etree.ElementTree as ET
import copy
import re
#from src import treadmill_constraint
from src.track_duration import Objective as TrackDurationObjective
from src.track_grf_contact import Objective as TrackGrfContactObjective
from src.track_angles_pooled_variance import install_as_biosym_track_angles
from src.pooled_angle_variance import add_pooled_variance_files
from src.contact_huntcrossley import install as install_huntcrossley_contact
from src.muscle_volume_weighted_effort import (
    select_effort_objective,
)
from biosym.utils import read_mot, write_mot
from biosym.utils.states import *
import numpy as np
import yaml
from src.gait_data import DEFAULT_GAIT_DATA_FILE, get_duration

# Parse the speed argument from the command line.
import argparse
import subprocess
import sys
install_as_biosym_track_angles()
parser = argparse.ArgumentParser(description='Track overground walking at a prescribed forward speed.')
parser.add_argument('--speed', type=float, default=None, help='Target forward walking speed, e.g. 1.2.')
parser.add_argument('--n-nodes', type=int, default=None, help='Override collocation node count, e.g. 150.')
parser.add_argument('--participant', type=int, default=None, help='Participant index for duration lookup and config file naming.')
parser.add_argument(
    '--model-dir',
    default=None,
    help='Optional model directory containing gait2d_scaled.yaml. Defaults to the participant model.',
)
parser.add_argument(
    '--model-file',
    default=None,
    help=(
        'Optional complete Biosym model YAML path. For Hunt-Crossley this '
        'defaults to the P1 Hunt-Crossley heel/toe-point model YAML.'
    ),
)
parser.add_argument(
    '--angle-reference-file',
    default=None,
    help='Explicit angle-tracking MOT file. Overrides the default Winter reference without changing the GRF source.',
)
parser.add_argument(
    '--grf-reference-file',
    default=None,
    help='Explicit experimental GRF MOT file. Defaults to the selected participant and speed.',
)
parser.add_argument(
    '--tracking-config',
    default=None,
    help='Walking tracking YAML file. Defaults to walking_tracking_p{participant}_{speed}.yaml.',
)
parser.add_argument(
    '--output-file',
    default=None,
    help='Override collocation.settings.output.file from the YAML.',
)
parser.add_argument(
    '--initial-guess-file',
    default=None,
    help='Override collocation.initial_guess.file from the YAML and use type=from_file.',
)
parser.add_argument(
    '--no-visualize',
    action='store_true',
    help='Do not show the walking visualization after solving.',
)
parser.add_argument(
    '--summary-file',
    default=None,
    help='Path for the saved summary plot. Defaults to the output file path with a .png suffix.',
)
parser.add_argument(
    '--no-auto-summary',
    action='store_true',
    help='Do not automatically save the walking summary plot after solving.',
)
parser.add_argument(
    '--validate-only',
    action='store_true',
    help='Build and compile the OCP, print the selected references, and exit without solving.',
)
parser.add_argument(
    '--live-dashboard',
    action='store_true',
    help='Show IPOPT objective convergence in a live browser dashboard while solving.',
)
parser.add_argument(
    '--dashboard-port',
    type=int,
    default=8050,
    help='Local port for the live solver dashboard (default: 8050).',
)
parser.add_argument(
    '--iteration-log-interval',
    type=int,
    default=5,
    help='Update the live dashboard every N IPOPT iterations (default: 5).',
)
parser.add_argument('--effort-model', choices=('activation', 'muscle_volume'), default='activation')
parser.add_argument('--muscle-effort-weight', type=float, default=None)
parser.add_argument(
    '--grf-force-threshold',
    type=float,
    default=None,
    help='Track only experimental GRF components with magnitude above this value in N.',
)
parser.add_argument(
    '--muscle-effort-speedweighting',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Divide effort by the fixed requested walking speed^exponent.',
)
args = parser.parse_args()

default_template_path = Path("walking_tracking_overground.yaml")
config_path = args.tracking_config or (
    str(default_template_path)
    if default_template_path.exists()
    else None
)
if config_path is None:
    if args.participant is None or args.speed is None:
        parser.error("Either pass --tracking-config, or pass --participant and --speed.")
    speed_string = f"{abs(float(args.speed)):.1f}".replace(".", "_")
    config_path = f"walking_tracking_p{args.participant}_{speed_string}.yaml"

with open(config_path, "r", encoding="utf-8") as handle:
    config_payload = yaml.safe_load(handle)

collocation_settings = copy.deepcopy(config_payload["collocation"])
is_template_config = Path(config_path).name in {
    "walking_tracking_template.yaml",
    "walking_tracking_overground.yaml",
}


def infer_participant_from_config(config_path, collocation_settings):
    match = re.search(r"p(\d+)", Path(config_path).name)
    if match:
        return int(match.group(1))
    for objective in collocation_settings.get("objectives", []):
        if objective.get("name") == "track_duration":
            person = objective.get("args", {}).get("person")
            if person is not None:
                return int(person)
    return None


def infer_speed_from_config(config_path, collocation_settings):
    match = re.search(r"p\d+_(\d+)_(\d+)\.ya?ml$", Path(config_path).name)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    for objective in collocation_settings.get("objectives", []):
        if objective.get("name") == "track_duration":
            speed = objective.get("args", {}).get("speed")
            if speed is not None:
                return float(speed)
    return None


participant = args.participant if args.participant is not None else infer_participant_from_config(config_path, collocation_settings)
speed = args.speed if args.speed is not None else infer_speed_from_config(config_path, collocation_settings)
if participant is None:
    parser.error("--participant is required when the tracking config does not define a participant.")
if speed is None:
    parser.error("--speed is required when the tracking config does not define a speed.")
if is_template_config and args.output_file is None:
    parser.error("--output-file is required when using a tracking template.")
if is_template_config and args.initial_guess_file is None:
    parser.error("--initial-guess-file is required when using a tracking template.")

participant = int(participant)
speed = float(speed)
target_speed = np.abs(speed)
install_huntcrossley_contact()
select_effort_objective(
    collocation_settings,
    args.effort_model,
    speed,
    args.muscle_effort_weight,
    args.muscle_effort_speedweighting,
)
model_dir = (
    Path(args.model_dir)
    if args.model_dir is not None
    else Path("models") / f"gait2d_scaled_p{participant}"
)
model_file = (
    Path(args.model_file)
    if args.model_file is not None
    else model_dir / "gait2d_scaled_huntcrossley_6spheres.yaml"
)
if not model_file.exists():
    raise FileNotFoundError(f"Biomechanical model configuration was not found: {model_file}")
print(f"Using biomechanical model from {model_file}")
m = model.load_model(str(model_file), force_rebuild=True)
model_mass = float(np.asarray(m.default_constants.mass).sum())
print(f"Model mass: {model_mass:.2f} kg")
contact_class = m.gc_model.__class__.__name__
is_huntcrossley_multi = (
    contact_class == "MultiContact"
    and all(type(contact).__name__ == "HuntCrossley" for contact in m.gc_model.models)
)
if contact_class != "HuntCrossley" and not is_huntcrossley_multi:
    raise ValueError(
        "The model YAML must reference Hunt-Crossley contact; "
        f"got {contact_class}."
    )
if is_huntcrossley_multi:
    groups = ", ".join(
        f"{len(contact.pairs)} contacts at E={contact.material_stiffness:g} Pa, "
        f"k_H={contact.hertz_coefficients[0]:g} N/m^(3/2)"
        for contact in m.gc_model.models
    )
    print(f"Using heterogeneous Hunt-Crossley contact groups: {groups}.")
else:
    print(
        "Using Biosym HuntCrossley fixed contacts: "
        f"E={m.gc_model.material_stiffness:g} Pa, "
        f"k_H={m.gc_model.hertz_coefficients[0]:g} N/m^(3/2), "
        f"dissipation={m.gc_model.c:g} s/m, "
        f"friction=({m.gc_model.us:g}, {m.gc_model.ud:g}, {m.gc_model.uv:g}), "
        f"transition_velocity={m.gc_model.v_t:g} m/s."
    )
print(
    "Contact evaluation path: Biosym model.run['gc_model']."
)
collocation_settings.setdefault("settings", {})["model"] = str(model_file)
if args.n_nodes is not None:
    collocation_settings.setdefault("settings", {})["nnodes"] = int(args.n_nodes)

nnodes = int(collocation_settings["settings"]["nnodes"])
if args.output_file is not None:
    collocation_settings.setdefault("settings", {}).setdefault("output", {})["file"] = args.output_file
if args.initial_guess_file is not None:
    collocation_settings["initial_guess"] = {
        "type": "from_file",
        "file": args.initial_guess_file,
    }
    print(
        "Starting overground tracking from standing solution: "
        f"{Path(args.initial_guess_file).resolve()}"
    )


def resample_periodic_columns(values, n_nodes):
    values = np.asarray(values, dtype=float)
    if values.shape[0] == n_nodes:
        return values.copy()

    old_nodes = values.shape[0]
    old_x = np.arange(old_nodes + 1, dtype=float) / old_nodes
    new_x = np.arange(n_nodes, dtype=float) / n_nodes
    periodic_values = np.vstack([values, values[:1]])

    resampled = np.empty((n_nodes, values.shape[1]), dtype=float)
    for col_idx in range(values.shape[1]):
        resampled[:, col_idx] = np.interp(new_x, old_x, periodic_values[:, col_idx])
    return resampled


def resample_presegmented_mot_file(mot_file, n_nodes):
    source_path = Path(mot_file)
    dataframe = read_mot(str(source_path))
    if dataframe.shape[0] == n_nodes:
        return str(source_path)

    resampled = dataframe.copy().iloc[:n_nodes].copy()
    resampled = resampled.reindex(range(n_nodes))

    value_columns = [column for column in dataframe.columns if column.lower() != "time"]
    resampled.loc[:, value_columns] = resample_periodic_columns(
        dataframe[value_columns].to_numpy(),
        n_nodes,
    )

    if "time" in dataframe.columns:
        start_time = float(dataframe["time"].iloc[0])
        stop_time = float(dataframe["time"].iloc[-1])
        resampled.loc[:, "time"] = np.linspace(start_time, stop_time, n_nodes)

    output_dir = Path("/tmp/ideal_treadmill_resampled_mot")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}_{n_nodes}_nodes{source_path.suffix}"
    write_mot(resampled, output_path, name=source_path.stem, in_degrees=False)
    print(f"Resampled {source_path} from {dataframe.shape[0]} to {n_nodes} rows: {output_path}")
    return str(output_path)


def speed_to_mot_token(speed):
    return f"{abs(float(speed)):.1f}".replace(".", "p")


def winter_angle_tracking_file(speed):
    cadence = "natural" if abs(float(speed)) <= 1.4 else "fast"
    path = Path(
        f"data/opensim_exports/winter_1987/"
        f"winter_1987_{cadence}_cadence_kinematics_mean_var_rad.mot"
    )
    print(f"Using Winter 1987 {cadence}-cadence joint-angle reference: {path}")
    return path


def update_tracking_case_settings(
    collocation_settings,
    participant,
    speed,
    angle_reference_file=None,
    grf_reference_file=None,
):
    grf_file = Path(f"data/opensim_exports/p{int(participant)}_grf_mean_speed_{speed_to_mot_token(speed)}.mot")
    if grf_reference_file:
        grf_file = Path(grf_reference_file)
    if angle_reference_file:
        angle_file = Path(angle_reference_file)
        print(f"Using explicit joint-angle reference: {angle_file}")
    else:
        angle_file = winter_angle_tracking_file(speed)
    if not grf_file.exists():
        raise FileNotFoundError(f"Expected GRF tracking file was not found: {grf_file}")

    for objective in collocation_settings.get("objectives", []):
        objective_args = objective.setdefault("args", {})
        if objective.get("name") == "track_grf":
            objective_args["presegmented"] = True
            objective_args["presegmented_file"] = str(grf_file)
            if args.grf_force_threshold is not None:
                objective_args["force_threshold"] = float(args.grf_force_threshold)
        elif objective.get("name") == "track_angles":
            if not angle_file.exists():
                raise FileNotFoundError(
                    f"Expected joint-angle tracking file was not found: {angle_file}"
                )
            objective_args["presegmented"] = True
            objective_args["presegmented_file"] = str(angle_file)
        elif objective.get("name") == "track_duration":
            objective_args["person"] = int(participant)
            objective_args["speed"] = abs(float(speed))
    return angle_file, grf_file


angle_tracking_file, grf_tracking_file = update_tracking_case_settings(
    collocation_settings,
    participant,
    speed,
    args.angle_reference_file,
    args.grf_reference_file,
)
add_pooled_variance_files(collocation_settings, angle_tracking_file)
# The built-in Biosym speed constraint acts on mean forward pelvis velocity.
# Fixing the global speed to the requested value produces overground progression.
collocation_settings.setdefault("bounds", {})["speed"] = [target_speed, target_speed]

for objective in collocation_settings.get("objectives", []):
    objective_args = objective.get("args", {})
    if objective_args.get("presegmented") and objective_args.get("presegmented_file"):
        objective_args["presegmented_file"] = resample_presegmented_mot_file(
            objective_args["presegmented_file"],
            nnodes,
        )

yaml_objectives = collocation_settings.get("objectives", [])
duration_objectives = [objective for objective in yaml_objectives if objective.get("name") == "track_duration"]
huntcrossley_grf_objectives = [objective for objective in yaml_objectives if objective.get("name") == "track_grf"]
collocation_settings["objectives"] = [
    objective
    for objective in yaml_objectives
    if objective.get("name") != "track_duration"
    and objective.get("name") != "track_grf"
]
for constraint in collocation_settings.get("constraints", []):
    if constraint.get("name") == "dynamics_unified":
        constraint["name"] = "dynamics"
col_obj = Collocation(m, collocation_settings)


print(f"Overground mode: target forward pelvis speed is {target_speed:g} m/s.")

#from src.track_angles_normalize_dur import Objective
#col_obj.objective.add_objective(Objective(col_obj.model, col_obj.settings, {
#    'presegmented': True,
#    'presegmented_file': "data/opensim_exports/kinematics_mean_var_rad.mot",
#    'exclude': ['q_pelvis_tx', 'q_pelvis_ty', 'q_pelvis_tilt'],
#}), weight=1.0)
for objective in huntcrossley_grf_objectives:
    if float(objective.get("weight", 1.0)) == 0.0:
        continue
    col_obj.objective.add_objective(
        TrackGrfContactObjective,
        objective.get("weight", 1.0),
        dict(objective.get("args", {})),
    )
for objective in duration_objectives:
    if float(objective.get("weight", 1.0)) == 0.0:
        continue
    duration_args = dict(objective.get("args", {}))
    col_obj.objective.add_objective(
        TrackDurationObjective,
        objective.get("weight", 1.0),
        duration_args,
    )

# These objectives are appended to the ObjectiveFunction after Collocation is
# constructed. Preserve their specifications in the serialized settings too,
# so downstream analyses can reconstruct every weighted objective term exactly.
col_obj.settings["objectives"].extend(
    [
        objective
        for objective in (
            huntcrossley_grf_objectives
            + duration_objectives
        )
        if float(objective.get("weight", 1.0)) != 0.0
    ]
)

# Match Biosym's documented standing-to-walking workflow: load the one-node
# standing solution through ``initial_guess: from_file`` and let Collocation
# repeat it over the walking nodes. Do not manufacture forward motion in the
# initial state; the speed constraint and excluded pelvis-translation
# periodicity coordinate establish progression during optimization.
col_obj.initial_guess_globals = col_obj.initial_guess_globals.replace(speed=target_speed)
print("Overground mode: stationary Hunt-Crossley half-space; unmodified standing seed.")
col_obj.setup()


def save_summary_plot(result_file, participant, speed, summary_file=None):
    result_path = Path(result_file)
    if summary_file is None:
        summary_file = result_path.with_suffix(".png")
    summary_path = Path(summary_file)
    speed_token = speed_to_mot_token(speed)
    desired_grf = Path(f"data/opensim_exports/p{int(participant)}_grf_mean_speed_{speed_token}.mot")
    command = [
        sys.executable,
        "visualize_walking2d.py",
        "--result",
        str(result_path),
        "--participant",
        str(int(participant)),
        "--speed",
        f"{float(speed):.1f}",
        "--desired-grf",
        str(desired_grf),
        "--save-summary",
        str(summary_path),
        "--no-stickfigure",
        "--no-show",
        "--reuse-model-cache",
    ]
    print("Saving summary plot:", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Warning: summary plot generation failed with exit code {exc.returncode}.")
    else:
        print(f"Saved summary plot to {summary_path}")


def save_summary_plot_from_solution(solution, biosym_model, participant, speed, summary_file=None, settings=None):
    from matplotlib import pyplot as plt
    from visualize_walking2d import (
        create_joint_angle_figure,
        create_slip_friction_figure,
        create_summary_figure,
        diagnose_foot_strike,
        foot_strike_label,
        get_treadmill_controller_metadata,
        joint_angle_output_path,
        load_desired_joint_angles,
        load_desired_belt_speed,
        load_desired_grf,
        resolve_joint_angle_reference_path,
        save_figure,
        slip_friction_output_path,
    )

    if len(solution) == 3:
        (states_dict, globals_dict), info, solution_settings = solution
        settings = solution_settings if settings is None else settings
    elif len(solution) == 2:
        (states_dict, globals_dict), info = solution
        if settings is None:
            settings = col_obj.settings
    else:
        raise ValueError(f"Unexpected solve result with {len(solution)} entries.")
    result_path = Path(col_obj.settings["settings"]["output"]["file"])
    if summary_file is None:
        summary_file = result_path.with_suffix(".png")
    summary_path = Path(summary_file)
    speed_token = speed_to_mot_token(speed)
    desired_grf_path = Path(f"data/opensim_exports/p{int(participant)}_grf_mean_speed_{speed_token}.mot")
    desired_grf = load_desired_grf(desired_grf_path, len(states_dict))
    desired_belt_speed = load_desired_belt_speed(
        str(DEFAULT_GAIT_DATA_FILE),
        int(participant),
        float(speed),
        len(states_dict),
    )
    joint_angle_reference_path = resolve_joint_angle_reference_path(settings, speed)
    desired_joint_angles = load_desired_joint_angles(
        joint_angle_reference_path,
        len(states_dict),
    )
    foot_strike_diagnosis = diagnose_foot_strike(biosym_model, states_dict)
    print(foot_strike_label(foot_strike_diagnosis))
    try:
        experimental_duration = get_duration(DEFAULT_GAIT_DATA_FILE, participant, speed)
    except (FileNotFoundError, ValueError):
        experimental_duration = None

    summary_fig = create_summary_figure(
        biosym_model,
        states_dict,
        globals_dict,
        desired_grf=desired_grf,
        desired_belt_speed=desired_belt_speed,
        experimental_duration=experimental_duration,
        treadmill_controller_metadata=get_treadmill_controller_metadata(settings),
        foot_strike_diagnosis=foot_strike_diagnosis,
    )
    save_figure(summary_fig, summary_path)
    plt.close(summary_fig)
    print(f"Saved summary plot to {summary_path}")
    joint_angle_path = joint_angle_output_path(summary_path)
    joint_angle_fig = create_joint_angle_figure(
        biosym_model,
        states_dict,
        desired_joint_angles=desired_joint_angles,
        foot_strike_diagnosis=foot_strike_diagnosis,
    )
    save_figure(joint_angle_fig, joint_angle_path)
    plt.close(joint_angle_fig)
    print(f"Saved joint-angle plot to {joint_angle_path}")
    slip_path = slip_friction_output_path(summary_path)
    slip_fig = create_slip_friction_figure(biosym_model, states_dict)
    save_figure(slip_fig, slip_path)
    plt.close(slip_fig)
    print(f"Saved slip/friction plot to {slip_path}")


def save_huntcrossley_tracking_plots(
    solution,
    biosym_model,
    angle_reference_file,
    grf_reference_file,
    summary_file=None,
):
    """Save Biosym 0.1.8 angle and GRF comparisons without legacy contact code."""
    from matplotlib import pyplot as plt
    from visualize_walking2d import (
        create_slip_friction_figure,
        slip_friction_output_path,
    )

    states = solution.states
    n_nodes = len(states)
    result_path = Path(col_obj.settings["settings"]["output"]["file"])
    grf_output = Path(summary_file) if summary_file else result_path.with_suffix(".png")
    angle_output = grf_output.with_name(f"{grf_output.stem}_joint_angles.png")
    slip_output = Path(slip_friction_output_path(grf_output))
    grf_output.parent.mkdir(parents=True, exist_ok=True)

    angle_data = read_mot(str(angle_reference_file))
    grf_data = read_mot(str(grf_reference_file))
    gait_cycle = np.linspace(0.0, 100.0, n_nodes)
    def resample_for_plot(values):
        values = np.asarray(values, dtype=float)
        if len(values) == n_nodes:
            return values
        old_phase = np.linspace(0.0, 1.0, len(values), endpoint=False)
        new_phase = np.linspace(0.0, 1.0, n_nodes, endpoint=False)
        return np.interp(
            new_phase,
            np.append(old_phase, 1.0),
            np.append(values, values[0]),
        )

    joint_rows = (
        (("q_hip_r", "hip_flexion_r_mean", 0, "Right hip"), ("q_hip_l", "hip_flexion_l_mean", 3, "Left hip")),
        (("q_knee_r", "knee_angle_r_mean", 1, "Right knee"), ("q_knee_l", "knee_angle_l_mean", 4, "Left knee")),
        (("q_ankle_r", "ankle_angle_r_mean", 2, "Right ankle"), ("q_ankle_l", "ankle_angle_l_mean", 5, "Left ankle")),
    )
    fig_angles, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True)
    q = np.asarray(states.q, dtype=float)
    moments = np.asarray(
        biosym_model.run["actuator_model"](states, biosym_model.default_constants),
        dtype=float,
    )
    for row, sides in enumerate(joint_rows):
        for side, (model_name, data_name, moment_index, title) in enumerate(sides):
            angle_axis = axes[row, side * 2]
            moment_axis = axes[row, side * 2 + 1]
            simulated = np.rad2deg(q[:, biosym_model.coordinates.names.index(model_name)])
            experimental = resample_for_plot(angle_data[data_name])
            variance_name = data_name.replace("_mean", "_var")
            experimental_variance = resample_for_plot(angle_data[variance_name])
            experimental = np.rad2deg(experimental)
            experimental_std = np.rad2deg(
                np.sqrt(np.maximum(experimental_variance, 0.0))
            )
            angle_axis.plot(gait_cycle, simulated, label="Optimized", linewidth=2)
            angle_axis.plot(gait_cycle, experimental, ":", label="Experimental mean", linewidth=2)
            angle_axis.fill_between(
                gait_cycle,
                experimental - experimental_std,
                experimental + experimental_std,
                alpha=0.2,
                label="Experimental ±1 SD",
            )
            angle_axis.set_title(f"{title} angle")
            angle_axis.set_ylabel("Angle (deg)")
            angle_axis.grid(True, alpha=0.3)
            moment_axis.plot(gait_cycle, moments[:, moment_index], linewidth=2)
            moment_axis.axhline(0.0, color="k", linewidth=0.7)
            moment_axis.set_title(f"{title} net moment")
            moment_axis.set_ylabel("Moment (N m)")
            moment_axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Gait cycle (%)")
    axes[0, 0].legend()
    fig_angles.tight_layout()
    fig_angles.savefig(angle_output, dpi=180, bbox_inches="tight")
    plt.close(fig_angles)

    evaluated_forces = []
    for node in range(n_nodes):
        state_with_forces = calc_forces(states[node], biosym_model)
        # Hunt-Crossley model order is right foot followed by left foot.
        evaluated_forces.append(
            np.asarray(state_with_forces.ext_forces, dtype=float).reshape(2, 3)
        )
    evaluated_forces = np.asarray(evaluated_forces)
    grf_pairs = (
        (0, 0, "grf_x_r_mean", "Right horizontal GRF"),
        (0, 1, "grf_y_r_mean", "Right vertical GRF"),
        (1, 0, "grf_x_l_mean", "Left horizontal GRF"),
        (1, 1, "grf_y_l_mean", "Left vertical GRF"),
    )
    fig_grf, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (foot, component, data_name, title) in zip(axes.flat, grf_pairs):
        axis.plot(gait_cycle, evaluated_forces[:, foot, component], label="Optimized", linewidth=2)
        axis.plot(
            gait_cycle,
            resample_for_plot(grf_data[data_name]),
            ":",
            label="Experimental",
            linewidth=2,
        )
        axis.set_title(title)
        axis.set_ylabel("Force (N)")
        axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Gait cycle (%)")
    axes[0, 0].legend()
    fig_grf.tight_layout()
    fig_grf.savefig(grf_output, dpi=180, bbox_inches="tight")
    plt.close(fig_grf)
    fig_slip = create_slip_friction_figure(biosym_model, states)
    fig_slip.savefig(slip_output, dpi=180, bbox_inches="tight")
    plt.close(fig_slip)
    print(f"Saved GRF comparison plot to {grf_output}")
    print(f"Saved joint-angle comparison plot to {angle_output}")
    print(f"Saved slip/friction plot to {slip_output}")

if args.live_dashboard:
    try:
        col_obj.enable_logging(
            log_interval=args.iteration_log_interval,
            launch_dashboard=True,
            dashboard_port=args.dashboard_port,
        )
        print(
            f"Live solver dashboard: http://localhost:{args.dashboard_port} "
            f"(updates every {args.iteration_log_interval} IPOPT iterations)"
        )
    except PermissionError:
        print(
            "Could not open the live solver dashboard port; "
            "continuing the optimization without it."
        )

if args.validate_only:
    objective_value = col_obj.objective.objfun(
        col_obj.initial_guess_states,
        col_obj.initial_guess_globals,
    )
    constraint_values = col_obj.constraints.confun(
        col_obj.initial_guess_states,
        col_obj.initial_guess_globals,
    )
    jax.block_until_ready(objective_value)
    jax.block_until_ready(constraint_values)
    constraint_values_np = np.asarray(constraint_values, dtype=float)
    print(
        f"Initial-guess objective={float(objective_value):.6g}; "
        f"constraints evaluated ({constraint_values.size} values); "
        f"maximum absolute residual={np.max(np.abs(constraint_values_np)):.6g}."
    )
    for constraint_index, constraint in enumerate(col_obj.constraints._constraints):
        start = col_obj.constraints.c_start[constraint_index]
        stop = col_obj.constraints.c_start[constraint_index + 1]
        name = constraint._get_info().get("name", type(constraint).__name__)
        values = constraint_values_np[start:stop]
        print(
            f"  {name}: max |residual|={np.max(np.abs(values)):.6g}, "
            f"RMS={np.sqrt(np.mean(values**2)):.6g} ({values.size} values)"
        )
    print("Validation complete; --validate-only requested, so the optimizer was not started.")
    sys.exit(0)

res = col_obj.solve(visualize=not args.no_visualize)

if not args.no_auto_summary:
    angle_plot_reference = args.angle_reference_file or winter_angle_tracking_file(speed)
    grf_plot_reference = args.grf_reference_file or Path(
        f"data/opensim_exports/p{int(participant)}_grf_mean_speed_{speed_to_mot_token(speed)}.mot"
    )
    save_huntcrossley_tracking_plots(
        res,
        col_obj.model,
        angle_plot_reference,
        grf_plot_reference,
        args.summary_file,
    )

# cyipopt/OpenSim can abort during C++ teardown after the result has already
# been written. The tracking script is run as a subprocess, so exit before
# those native destructors run and let the pipeline continue cleanly.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
