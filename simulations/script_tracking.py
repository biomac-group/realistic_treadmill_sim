import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import os
import csv
import cloudpickle
from pathlib import Path

from biosym.ocp.collocation import Collocation
from biosym.model import model
from biosym.model.contact import contact_parser
import xml.etree.ElementTree as ET
import copy
import re
from src.gc_model_huntcrossley_treadmill import (
    HuntCrossleyTreadmill,
    MultiHuntCrossleyTreadmill,
)
from src.contact_huntcrossley import install as install_huntcrossley_contact
#from src import treadmill_constraint
from src import treadmill_constraint_swing_target as treadmill_constraint
from src.track_duration import Objective as TrackDurationObjective
from src.track_grf_contact import Objective as TrackGrfContactObjective
from src.track_angles_pooled_variance import install_as_biosym_track_angles
from src.pooled_angle_variance import add_pooled_variance_files
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
import cloudpickle
import subprocess
import sys
install_as_biosym_track_angles()
parser = argparse.ArgumentParser(description='Track treadmill speed in a walking solution.')
parser.add_argument('--speed', type=float, default=None, help='Target treadmill speed for tracking, e.g. 1.2.')
parser.add_argument('--n-nodes', type=int, default=None, help='Override collocation node count, e.g. 150.')
parser.add_argument('--participant', type=int, default=None, help='Participant index for duration lookup and config file naming.')
parser.add_argument('--live-dashboard', action='store_true', help='Launch the Biosym/IPOPT live solver dashboard.')
parser.add_argument('--dashboard-port', type=int, default=8050, help='Port for the live solver dashboard.')
parser.add_argument('--iteration-log-interval', type=int, default=5, help='Log every N IPOPT iterations.')
parser.add_argument('--effort-model', choices=('activation', 'muscle_volume'), default='activation')
parser.add_argument('--muscle-effort-weight', type=float, default=None)
parser.add_argument(
    '--grf-force-threshold',
    type=float,
    default=None,
    help='Track only experimental GRF components with magnitude above this value in N.',
)
parser.add_argument(
    '--grf-weight',
    type=float,
    default=None,
    help='Override the tracking YAML weight for the track_grf objective.',
)
parser.add_argument(
    '--muscle-effort-speedweighting',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Divide effort by the fixed requested walking speed^exponent.',
)
parser.add_argument(
    '--model-dir',
    default=None,
    help='Optional model directory containing gait2d_scaled.yaml. Defaults to the participant model.',
)
parser.add_argument(
    '--model-file',
    default=None,
    help='Explicit Biosym model YAML; required for Hunt-Crossley sphere conditions.',
)
parser.add_argument(
    '--reference-duration',
    type=float,
    default=None,
    help='Nominal gait-cycle duration used only to convert controller delays to samples.',
)
parser.add_argument(
    '--angle-reference-file',
    default=None,
    help='Explicit angle-tracking MOT file. Overrides the default Winter reference without changing the GRF source.',
)
parser.add_argument(
    '--grf-reference-file',
    default=None,
    help='Explicit GRF-tracking MOT file. Overrides the participant-specific default.',
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
    '--controller-config',
    default=None,
    help='Optional JSON file containing per-node controller timing settings.',
)
parser.add_argument(
    '--controller-params-json',
    default='ocp_controller_params.json',
    help='Controller parameter JSON for the OCP. Can be keyed by node count or be a fitted result JSON.',
)
parser.add_argument('--delay-seconds', type=float, default=None, help='Manual GRF delay in seconds.')
parser.add_argument('--k-fy', type=float, default=None, help='Manual vertical GRF gain.')
parser.add_argument('--k-fx', type=float, default=None, help='Manual horizontal GRF gain.')
parser.add_argument('--k-p', type=float, default=None, help='Manual proportional gain.')
parser.add_argument('--k-d', type=float, default=None, help='Manual derivative gain.')
parser.add_argument('--error-delay-samples', type=int, default=None, help='Manual PD error delay in samples.')
parser.add_argument(
    '--no-post-hs-interpolation',
    action='store_true',
    help='Use the raw AP contact force in the treadmill controller without the post-heel-strike cubic bridge.',
)
parser.add_argument(
    '--controller-edge-source-file',
    default=None,
    help='Ideal-treadmill contact diagnostics CSV used to detect fixed per-side HS and AP-gap endpoints.',
)
parser.add_argument(
    '--controller-edge-source-result',
    default=None,
    help='Original overground result PKL used for stationary-contact HS and AP-edge detection.',
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
    '--summary-from-memory',
    action='store_true',
    help='Create the summary plot directly from the solved OCP result instead of reloading the saved pickle.',
)
parser.add_argument(
    '--validate-only',
    action='store_true',
    help='Compile and evaluate the treadmill OCP without starting IPOPT.',
)
args = parser.parse_args()

default_template_path = Path("walking_tracking_template.yaml")
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
is_template_config = Path(config_path).name == "walking_tracking_template.yaml"


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
    parser.error("--output-file is required when using walking_tracking_template.yaml.")
if is_template_config and args.initial_guess_file is None:
    parser.error("--initial-guess-file is required when using walking_tracking_template.yaml.")

participant = int(participant)
speed = float(speed)
target_speed = -np.abs(speed)
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
    if args.model_file
    else model_dir / "gait2d_scaled_huntcrossley_6spheres.yaml"
)
if not model_file.exists():
    raise FileNotFoundError(f"Biomechanical model configuration was not found: {model_file}")
print(f"Using biomechanical model from {model_file}")
m = model.load_model(str(model_file), force_rebuild=True)
model_mass = float(np.asarray(m.default_constants.mass).sum())
print(f"Model mass: {model_mass:.2f} kg")
if m.gc_model.__class__.__name__ == 'HuntCrossley':
    m.gc_model = HuntCrossleyTreadmill(m.gc_model)
    contact_description = "six fixed spheres per foot"
elif m.gc_model.__class__.__name__ == 'MultiContact':
    m.gc_model = MultiHuntCrossleyTreadmill(m.gc_model)
    contact_description = "six fixed spheres per foot"
else:
    raise ValueError(f"Expected Hunt-Crossley contact; got {type(m.gc_model).__name__}.")
m.gc_model.process_eom(m)
m._register_contact_model(m.gc_model)
print(
    f"Treadmill contact: Hunt-Crossley {contact_description}; "
    "gc_model states are right/left belt surface velocities."
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
    source_guess = Path(args.initial_guess_file)
    with source_guess.open("rb") as handle:
        (guess_states, guess_globals), guess_info, guess_settings = cloudpickle.load(handle)
    gc_values = np.asarray(guess_states.gc_model)
    if gc_values.shape[-1] != 2:
        adapted_states = guess_states.replace(
            gc_model=jnp.full((len(guess_states), 2), target_speed)
        )
        adapted_path = (
            Path("/tmp/ideal_treadmill_biosym_initial_guesses")
            / f"{source_guess.stem}_huntcrossley_treadmill.pkl"
        )
        adapted_path.parent.mkdir(parents=True, exist_ok=True)
        with adapted_path.open("wb") as handle:
            cloudpickle.dump(
                ((adapted_states, guess_globals), guess_info, guess_settings),
                handle,
            )
        collocation_settings["initial_guess"]["file"] = str(adapted_path)
        print(f"Added right/left belt-speed states to standing guess: {adapted_path}")


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
    grf_file = (
        Path(grf_reference_file)
        if grf_reference_file
        else Path(
            f"data/opensim_exports/p{int(participant)}_grf_mean_speed_"
            f"{speed_to_mot_token(speed)}.mot"
        )
    )
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
            if args.grf_weight is not None:
                if args.grf_weight < 0.0:
                    raise ValueError("--grf-weight must be non-negative.")
                objective["weight"] = float(args.grf_weight)
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



def detect_controller_interpolation_windows(model, states, globals_, nominal_samples):
    """Detect fixed per-side HS and AP-gap endpoints from the loaded seed."""
    local_states = states[:nnodes]
    if local_states.qdd is None:
        qd = np.asarray(local_states.qd, dtype=float)
        duration = float(np.asarray(globals_.dur))
        dt = duration / nnodes
        qdd = np.empty_like(qd)
        qdd[1:] = np.diff(qd, axis=0) / dt
        qdd[0] = (qd[0] - qd[-1]) / dt
        local_states = local_states.replace(qdd=jnp.asarray(qdd))

    def contact_force(state):
        full_state = state.replace(
            tau=jnp.zeros(model.tau.n),
            ext_forces=jnp.zeros(model.ext_forces.n),
            ext_torques=jnp.zeros(model.ext_torques.n),
        )
        return model.run["gc_model"](full_state, model.default_constants)[0]

    forces = np.asarray(jax.vmap(contact_force)(local_states))

    def detect_side(fx, fy):
        contact = fy >= 10.0
        crossings = np.flatnonzero(contact & ~np.roll(contact, 1))
        if not crossings.size:
            raise ValueError("No sustained vertical-GRF contact onset was detected.")
        scores = [np.sum(fy[(node + np.arange(15)) % nnodes]) for node in crossings]
        threshold_crossing = int(crossings[int(np.argmax(scores))])
        heel_strike = (threshold_crossing - 1) % nnodes

        # Search for rebound edges through post-HS sample 10. Sample 11 is
        # included so a candidate maximum at sample 10 can be confirmed by
        # the following decline.
        offsets = np.arange(max(int(nominal_samples) + 4, 12))
        local_fx = fx[(heel_strike + offsets) % nnodes]
        slopes = np.diff(local_fx)
        apexes = np.flatnonzero((slopes[:-1] > 0.0) & (slopes[1:] < 0.0)) + 1
        if not apexes.size:
            print(
                "Warning: no post-HS AP rebound edge was detected; "
                "disabling post-HS interpolation for this side."
            )
            return heel_strike, 0
        # Select the sharpest positive-to-negative reversal, rather than the
        # first local maximum, which may only be a small post-HS oscillation.
        slope_reversals = slopes[apexes - 1] - slopes[apexes]
        unchanged_edge_offset = int(apexes[np.argmax(slope_reversals)])
        return heel_strike, unchanged_edge_offset

    hs_r, samples_r = detect_side(forces[:, 0, 0], forces[:, 0, 1])
    hs_l, samples_l = detect_side(forces[:, 1, 0], forces[:, 1, 1])
    return {
        "heel_strike_index_r": hs_r,
        "heel_strike_index_l": hs_l,
        "post_hs_interpolation_samples_r": samples_r,
        "post_hs_interpolation_samples_l": samples_l,
    }

def load_controller_settings(config_path, params_json, cli_args, n_nodes, participant, speed):
    node_key = str(int(n_nodes))
    node_config = {}
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with config_file.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
            node_config = payload.get(node_key)
            if node_config is None:
                fallback_key = "100" if "100" in payload else next(iter(payload))
                node_config = payload[fallback_key]
                print(
                    f"Warning: {config_path} has no {node_key!r} controller timing block; "
                    f"using {fallback_key!r} for error delay settings."
                )
        else:
            print(f"Warning: controller timing config {config_path!r} was not found; using defaults.")
    params = node_config.get("initial", node_config)
    if params_json:
        with open(params_json, "r", encoding="utf-8") as handle:
            result_payload = yaml.safe_load(handle)
        if node_key in result_payload:
            params = result_payload[node_key]
        elif "params" in result_payload:
            params = result_payload["params"]
        elif result_payload and all(isinstance(value, dict) for value in result_payload.values()):
            fallback_key = "100" if "100" in result_payload else next(iter(result_payload))
            params = result_payload[fallback_key]
            print(
                f"Warning: controller params file has no {node_key!r} block; "
                f"using {fallback_key!r}."
            )
        else:
            params = result_payload
        if "params" in params:
            params = params["params"]
        if int(result_payload.get("n_nodes", n_nodes)) != int(n_nodes):
            print(
                f"Warning: controller params were saved for n_nodes={result_payload.get('n_nodes')}, "
                f"but this OCP run uses n_nodes={n_nodes}."
            )
        node_config = {**node_config, **params}

    manual_params = {
        "delay_seconds": cli_args.delay_seconds,
        "k_fy": cli_args.k_fy,
        "k_fx": cli_args.k_fx,
        "k_p": cli_args.k_p,
        "k_d": cli_args.k_d,
    }
    params = {**params, **{key: value for key, value in manual_params.items() if value is not None}}

    if cli_args.error_delay_samples is not None:
        error_delay_samples = int(cli_args.error_delay_samples)
    else:
        error_delay_samples = int(node_config.get("error_delay_samples", 1))

    duration = (
        float(cli_args.reference_duration)
        if cli_args.reference_duration is not None
        else get_duration(DEFAULT_GAIT_DATA_FILE, participant, speed)
    )
    dt = duration / float(n_nodes)
    delay = round(float(params["delay_seconds"]) / dt)
    print("delay of", delay)

    return {
        "delay": delay,
        "error_delay_samples": error_delay_samples,
        "k_fy": float(params["k_fy"]),
        "k_fx": float(params["k_fx"]),
        "k_p": float(params["k_p"]),
        "k_d": float(params["k_d"]),
        "post_hs_interpolation_samples": (
            0 if cli_args.no_post_hs_interpolation else max(1, round(0.050 / dt))
        ),
        "ap_grf_filter_enabled": False,
    }


controller_settings = load_controller_settings(
    args.controller_config,
    args.controller_params_json,
    args,
    nnodes,
    participant,
    speed,
)
if args.no_post_hs_interpolation:
    controller_settings.update({
        "post_hs_interpolation_samples": 0,
        "post_hs_interpolation_samples_r": 0,
        "post_hs_interpolation_samples_l": 0,
    })
elif args.controller_edge_source_result:
    with open(args.controller_edge_source_result, "rb") as handle:
        (source_states, source_globals), _, source_settings = cloudpickle.load(handle)
    source_model_file = source_settings.get("settings", {}).get("model")
    if not source_model_file:
        raise ValueError(
            f"No model path stored in {args.controller_edge_source_result}"
        )
    # Load a separate, stationary-ground model. Do not convert this instance
    # to treadmill contact: detection must use the original overground forces.
    source_model = model.load_model(source_model_file, force_rebuild=False)
    detected = detect_controller_interpolation_windows(
        source_model,
        source_states,
        source_globals,
        controller_settings["post_hs_interpolation_samples"],
    )
    detected["controller_edge_source_result"] = args.controller_edge_source_result
    controller_settings.update(detected)
elif args.controller_edge_source_file:
    grouped = {
        "right": np.zeros((nnodes, 2)),
        "left": np.zeros((nnodes, 2)),
    }
    with open(args.controller_edge_source_file, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            node = int(row["node"])
            if node >= nnodes:
                continue
            side = row["side"]
            grouped[side][node, 0] += float(row["friction_force_x_N"])
            grouped[side][node, 1] += float(row["normal_force_N"])

    def detect_saved_side(values):
        fx, fy = values.T
        contact = fy >= 10.0
        crossings = np.flatnonzero(contact & ~np.roll(contact, 1))
        scores = [np.sum(fy[(node + np.arange(15)) % nnodes]) for node in crossings]
        crossing = int(crossings[int(np.argmax(scores))])
        hs = (crossing - 1) % nnodes
        # Search for rebound edges through post-HS sample 10. Sample 11 is
        # included so a candidate maximum at sample 10 can be confirmed by
        # the following decline.
        offsets = np.arange(max(
            controller_settings["post_hs_interpolation_samples"] + 4, 12
        ))
        local_fx = fx[(hs + offsets) % nnodes]
        slopes = np.diff(local_fx)
        apexes = np.flatnonzero((slopes[:-1] > 0) & (slopes[1:] < 0)) + 1
        if not apexes.size:
            print(
                "Warning: no post-HS AP edge detected in "
                f"{args.controller_edge_source_file}; disabling post-HS "
                "interpolation for this side."
            )
            return hs, 0
        # Select the local maximum with the strongest slope reversal.
        slope_reversals = slopes[apexes - 1] - slopes[apexes]
        return hs, int(apexes[np.argmax(slope_reversals)])

    hs_r, length_r = detect_saved_side(grouped["right"])
    hs_l, length_l = detect_saved_side(grouped["left"])
    controller_settings.update({
        "heel_strike_index_r": hs_r,
        "heel_strike_index_l": hs_l,
        "post_hs_interpolation_samples_r": length_r,
        "post_hs_interpolation_samples_l": length_l,
        "controller_edge_source_file": args.controller_edge_source_file,
    })
print(
    "Treadmill controller settings:",
    {
        **controller_settings,
        "n_nodes": nnodes,
    },
)
treadmill_constraint_args = {
    **controller_settings,
    "target_speed": target_speed,
    "swing_force_threshold": 10.0,
}
col_obj.settings["treadmill_controller"] = dict(treadmill_constraint_args)
col_obj.constraints.add_constraint(
    treadmill_constraint.Constraint(col_obj.model, col_obj.settings, treadmill_constraint_args),
    weight=30,
    args=treadmill_constraint_args,
)

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
    [objective for objective in huntcrossley_grf_objectives if float(objective.get("weight", 1.0)) != 0.0]
    + duration_objectives
)

# Restore the cumulative totals after appending one additional constraint. This needs fixing!
constraint_infos = [constraint._get_info() for constraint in col_obj.constraints._constraints]
col_obj.constraints.c_start = np.cumsum([0] + [info["ncons"] for info in constraint_infos], dtype=np.int32).tolist()
col_obj.constraints.nnz_start = np.cumsum([0] + [info["nnz"] for info in constraint_infos], dtype=np.int32).tolist()
col_obj.constraints.ncon = col_obj.constraints.c_start[-1]
col_obj.constraints.nnz = col_obj.constraints.nnz_start[-1]
col_obj.constraints._compile_callables()

## Actually, biosym really struggles with user-defined contact models, which needs a bit of refactoring to work properly. 
# Re-setup the problem to account for the model changes
# Need to update the bounds for the new contact model. The tracking pipeline
# passes four zero gains for its fixed-treadmill stages, so pin both belt-speed
# states to the target speed in those stages. In the real-controller stage the
# belt speeds remain free within their normal operating range.
controller_is_zero = all(
    controller_settings[key] == 0.0
    for key in ("k_fy", "k_fx", "k_p", "k_d")
)
n_treadmill_nodes = col_obj.settings['nnodes_dur']
if controller_is_zero:
    min_treadmill_speed = np.full((n_treadmill_nodes, 2), target_speed)
    max_treadmill_speed = min_treadmill_speed.copy()
    print(f"Fixed treadmill mode: both belt speeds are fixed at {target_speed} m/s.")
else:
    min_treadmill_speed = np.full((n_treadmill_nodes, 2), -2.5)
    max_treadmill_speed = np.full((n_treadmill_nodes, 2), -0.5)
col_obj.settings['bounds']['min'] = col_obj.settings['bounds']['min'].replace(
    gc_model=jnp.asarray(min_treadmill_speed)
)
col_obj.settings['bounds']['max'] = col_obj.settings['bounds']['max'].replace(
    gc_model=jnp.asarray(max_treadmill_speed)
)
col_obj.setup()


def save_summary_plot(result_file, participant, speed, summary_file=None):
    result_path = Path(result_file)
    if summary_file is None:
        summary_file = result_path.with_suffix(".png")
    summary_path = Path(summary_file)
    desired_grf = grf_tracking_file
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
        save_contact_diagnostics_csv,
        slip_friction_output_path,
        contact_diagnostics_output_path,
    )

    if all(hasattr(solution, field) for field in ("states", "globals", "info")):
        # Return value of Collocation.solve(): Solution(states, globals, info).
        states_dict = solution.states
        globals_dict = solution.globals
        info = solution.info
        if settings is None:
            settings = col_obj.settings
    elif len(solution) == 3:
        # Serialized result: ((states, globals), info, settings).
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
    desired_grf_path = grf_tracking_file
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
    diagnostics_path = contact_diagnostics_output_path(result_path)
    save_contact_diagnostics_csv(biosym_model, states_dict, diagnostics_path)
    print(f"Saved contact diagnostics table to {diagnostics_path}")

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
        col_obj.initial_guess_states, col_obj.initial_guess_globals
    )
    constraint_values = col_obj.constraints.confun(
        col_obj.initial_guess_states, col_obj.initial_guess_globals
    )
    jax.block_until_ready(objective_value)
    jax.block_until_ready(constraint_values)
    print(
        f"Treadmill validation: objective={float(objective_value):.6g}, "
        f"max constraint residual={float(jnp.max(jnp.abs(constraint_values))):.6g}."
    )
    sys.exit(0)

res = col_obj.solve(visualize=not args.no_visualize)

if not args.no_auto_summary:
    if args.summary_from_memory:
        save_summary_plot_from_solution(
            res,
            col_obj.model,
            participant,
            speed,
            args.summary_file,
            settings=col_obj.settings,
        )
    else:
        save_summary_plot(
            col_obj.settings["settings"]["output"]["file"],
            participant,
            speed,
            args.summary_file,
        )

# cyipopt/OpenSim can abort during C++ teardown after the result has already
# been written. The tracking script is run as a subprocess, so exit before
# those native destructors run and let the pipeline continue cleanly.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
